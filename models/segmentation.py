import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim import SGD
from tqdm import tqdm
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, LovaszLoss

# import custom modules
from utils.utils import remove_small_regions_batch
from config.config_manager import ConfigManager


def train_segmentation(
    ssa_loader: torch.utils.data.DataLoader,
    hp_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    cfg: ConfigManager,
    device: torch.device,
    logger=None
):
    """
    Train segmentation with:
      - Dice + Lovász (masked by tissue)
      - Slide-level classification head (low weight)
      - Progressive introduction of HP (negatives) up to p_hp=0.3
      - Validation masked by tissue, threshold fixed at 0.5
      - Logging of PR AUC and mean IoU on validation
      - Save model whenever mean IoU at 0.5 improves
    """
    # Model initialization
    model = smp.DeepLabV3Plus(
        encoder_name=cfg.get("model.backbone"),
        encoder_weights="imagenet" if cfg.get("model.pretrained_backbone") else None,
        in_channels=3,
        classes=1,
        activation=None,
        decoder_atrous_rates=cfg.get("model.aspp_dilate", [6, 12, 18]),
    ).to(device)

    # Loss functions
    dice_loss_f = DiceLoss(mode="binary").to(device)
    lovasz_loss_f = LovaszLoss(mode="binary").to(device)

    # Optimizer
    optimizer = SGD(
        model.parameters(),
        lr=cfg.get("training.learning_rate", 5e-4),
        momentum=0.9,
        weight_decay=cfg.get("training.weight_decay", 1e-4),
    )
    steps_per_epoch = len(ssa_loader) + len(hp_loader)

    # optionally use a learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #    optimizer,
    #    T_max=cfg.get("training.epochs"),
    #    eta_min=cfg.get("training.min_lr", 1e-5),
    #    last_epoch=-1,
    # )

    scaler = GradScaler()

    warmup_ep = cfg.get("segmentation.train.warmup_epochs",
                        max(1, cfg.get("training.epochs") // 3))
    val_thresh = 0.5
    min_region = cfg.get("postprocessing.min_region", 500)
    cls_weight = cfg.get("segmentation.cls_loss_weight", 0.2)

    # IoU tracking across epochs
    best_iou_so_far = 0.0

    for epoch in range(1, cfg.get("training.epochs") + 1):
        model.train()
        epoch_loss = 0.0

        # Progressive learning with probability p_hp of choosing HP data
        # p_hp: 0 during warmup, then linearly up to a specified probability (0.3) over the next warmup_ep epochs
        if epoch <= warmup_ep:
            p_hp = 0.0
        else:
            frac = (epoch - warmup_ep) / warmup_ep
            p_hp = min(0.3, frac * 0.3)

        ssa_iter = iter(ssa_loader)
        hp_iter = iter(hp_loader)

        # Training loop
        for _ in tqdm(
            range(steps_per_epoch),
            desc=f"[Seg] Ep{epoch}/{cfg.get('training.epochs')}",
            ncols=80
        ):
            # Choose HP with probability p_hp, otherwise SSA
            if torch.rand(1).item() < p_hp:
                try:
                    imgs, gts, tissues = next(hp_iter)
                except StopIteration:
                    hp_iter = iter(hp_loader)
                    imgs, gts, tissues = next(hp_iter)
            else:
                try:
                    imgs, gts, tissues = next(ssa_iter)
                except StopIteration:
                    ssa_iter = iter(ssa_loader)
                    imgs, gts, tissues = next(ssa_iter)

            imgs    = imgs.to(device, non_blocking=True)     # (B,3,H,W)
            gts     = gts.to(device, non_blocking=True)      # (B,1,H,W) 0/1
            tissues = tissues.to(device, non_blocking=True)  # (B,1,H,W) 0/1

            optimizer.zero_grad()
            with autocast(device_type=device.type):
                seg_logits = model(imgs)            # (B,1,H,W)
                probs      = torch.sigmoid(seg_logits)

                B, C, H, W = seg_logits.shape
                # Flatten
                flat_logits = seg_logits.view(B, -1)    # (B, H*W)
                flat_probs  = probs.view(B, -1)         # (B, H*W)
                flat_gts    = gts.view(B, -1)           # (B, H*W)
                flat_tissue = tissues.view(B, -1)       # (B, H*W)

                # Masked Dice (only inside tissue)
                masked_probs = (flat_probs * flat_tissue).view(B, 1, H, W)
                masked_gts   = (flat_gts   * flat_tissue).view(B, 1, H, W)
                loss_dice    = dice_loss_f(masked_probs, masked_gts)

                # Masked Lovász (zero out logits outside tissue)
                logits_masked = (flat_logits * flat_tissue).view(B, 1, H, W)
                loss_lovasz   = lovasz_loss_f(logits_masked, masked_gts)

                seg_loss = loss_dice + 0.5 * loss_lovasz

                # Slide-level classification head (max over tissue)
                masked_logits = flat_logits * flat_tissue   # (B, H*W), zeros outside tissue
                cls_log = masked_logits.max(dim=1).values  # (B,)
                slide_lb = (masked_gts.view(B, -1).sum(dim=1) > 0).float().to(device)
                cls_loss = F.binary_cross_entropy_with_logits(cls_log, slide_lb)

                loss = seg_loss + cls_weight * cls_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        avg_train = epoch_loss / steps_per_epoch

        #scheduler.step()

        # swap learning rate if using a scheduler
        current_lr = 0.001
        #current_lr = optimizer.param_groups[0]["lr"]

        print(f"[Seg] Epoch {epoch} Train Loss: {avg_train:.4f} LR: {current_lr:.6f} p_hp={p_hp:.2f}")

        if logger:
            logger.log_scalar("Train/SegmLoss", avg_train, step=epoch)
            logger.log_scalar("Train/Segm_p_hp", p_hp, step=epoch)

        # Validation at fixed threshold 0.5
        model.eval()
        per_image_ious = []

        with torch.no_grad():
            for imgs, gts_b, tissues_b in val_loader:
                imgs      = imgs.to(device, non_blocking=True)       # (B,3,H,W)
                gts_b     = gts_b.to(device, non_blocking=True)      # (B,1,H,W)
                tissues_b = tissues_b.to(device, non_blocking=True)  # (B,1,H,W)

                with autocast(device_type=device.type):
                    logits = model(imgs)               # (B,1,H,W)
                    probs  = torch.sigmoid(logits)     # (B,1,H,W)

                B, C, H, W = probs.shape
                flat_probs_b  = probs.view(B, -1).cpu().numpy() # (B, H*W)
                flat_labels_b = gts_b.view(B, -1).cpu().numpy().astype(np.uint8)
                flat_tissue_b = tissues_b.view(B, -1).cpu().numpy().astype(np.uint8)

                for i in range(B):
                    mask_i = flat_tissue_b[i].astype(bool)
                    if mask_i.sum() == 0:
                        # skip this image if no tissue pixels
                        continue

                    # Per-image IoU (only on SSA slides: where GT sum>0 inside tissue)
                    l_i = flat_labels_b[i][mask_i]
                    if l_i.sum() == 0:
                        # HP (no lesion) - skip IoU
                        continue

                    p_i = flat_probs_b[i][mask_i]  # probabilities inside tissue
                    pred_bin_tissue = (p_i > val_thresh).astype(np.uint8)

                    # Reconstruct full (H×W) mask
                    full_pred_flat = np.zeros((H * W,), dtype=np.uint8)
                    full_pred_flat[mask_i] = pred_bin_tissue
                    full_pred = full_pred_flat.reshape(H, W)       # (H, W)

                    # Convert to 4D tensors (1,1,H,W) for small-region removal
                    tmp_pred   = torch.from_numpy(full_pred).unsqueeze(0).unsqueeze(0).float().to(device)
                    tmp_tissue = tissues_b[i, 0].unsqueeze(0).unsqueeze(0).float().to(device)

                    clean_pred_tensor = remove_small_regions_batch(tmp_pred, tmp_tissue, min_region)
                    clean_flat = clean_pred_tensor.view(-1).cpu().numpy().astype(np.uint8)[mask_i]

                    # Compute IoU inside tissue
                    inter = (clean_flat * l_i).sum()
                    union = ((clean_flat + l_i) >= 1).sum()
                    per_image_ious.append(inter / (union + 1e-6))

        # Compute mean IoU at a fixed threshold (0.5)
        if per_image_ious:
            mean_iou = float(np.mean(per_image_ious))
            std_iou  = float(np.std(per_image_ious))
            n_iou    = len(per_image_ious)
        else:
            mean_iou = 0.0
            std_iou  = 0.0
            n_iou    = 0

        print(f"[Seg] Validation @th={val_thresh:.2f}: mean IoU={mean_iou:.4f}, std={std_iou:.4f}, n={n_iou}")

        if logger:
            logger.log_scalar("Eval/Segm_IoU", mean_iou, step=epoch)

        # Model saving
        if mean_iou > best_iou_so_far:
            best_iou_so_far = mean_iou
            torch.save(model.state_dict(), cfg.get("model.save_path"))
            
            print(f"[Seg] Saved new best model @ Epoch {epoch}: IoU={best_iou_so_far:.4f} (th={val_thresh:.2f})")

    return model