import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet34_Weights

# import custom modules
from config.config_manager import ConfigManager
from utils.utils import ClassifierFocalTverskyLoss

def train_classifier(
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: ConfigManager,
        device: torch.device,
        logger=None):
    """
    Train classification backbone given pre-built train_loader and val_loader.
    Returns best_model_state_dict.
    """
    # Model initialization
    backbone_name = cfg.get("classification.backbone")
    backbone_cls = getattr(models, backbone_name)
    backbone = backbone_cls(weights=ResNet34_Weights.DEFAULT if backbone_name == "resnet34" else 'DEFAULT')
    backbone.fc = nn.Sequential(
       nn.Linear(backbone.fc.in_features, 128),
       nn.ReLU(inplace=True),
       nn.Dropout(0.5),
       nn.Linear(128, 1),
    )
    model = backbone.to(device)

    # Loss functions
    bce_loss = nn.BCEWithLogitsLoss()
    ft_loss = ClassifierFocalTverskyLoss(
        alpha=cfg.get("classification.tversky_alpha"),
        beta=cfg.get("classification.tversky_beta"),
        gamma=cfg.get("classification.tversky_gamma"),
    ).to(device)

    def loss_fn(logits, labels):
        labels_f = labels.float().unsqueeze(1).to(device)
        return bce_loss(logits, labels_f) + ft_loss(logits, labels_f)

    optimizer = SGD(
        model.parameters(),
        lr=cfg.get("classification.lr"),
        weight_decay=cfg.get("classification.weight_decay"),
        momentum=0.9,
    )

    total_epochs = sum(phase["epochs"] for phase in cfg.get("classification.unfreeze_schedule"))
    total_steps = total_epochs * len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg.get("classification.lr"),
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="linear",
    )

    best_val_acc = 0.0
    fixed_thresh = cfg.get("classification.threshold")
    epoch_counter = 0

    for phase_idx, phase in enumerate(cfg.get("classification.unfreeze_schedule")):
        layers_to_unfreeze = phase["layers"]
        epochs = phase["epochs"]

        # Freeze all layers
        for name, param in model.named_parameters():
            param.requires_grad = False
        # Unfreeze head + specified layers
        for name, param in model.named_parameters():
            if name.startswith("fc.") or any(name.startswith(f"layer{li}") for li in layers_to_unfreeze):
                param.requires_grad = True

        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Classifier] Phase {phase_idx+1}: unfreezing {layers_to_unfreeze}, {n_trainable} trainable params")

        for _ in range(epochs):
            epoch_counter += 1
            model.train()
            running_loss = 0.0
            running_acc = 0.0

            for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch_counter}/{total_epochs}", ncols=80):
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(imgs)
                loss = loss_fn(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    preds = (probs > fixed_thresh).float()
                    lbls = labels.float().unsqueeze(1)
                    running_acc += (preds == lbls).float().mean().item()
                running_loss += loss.item()

            n_batches = len(train_loader)
            print(f"[Clf] Loss={running_loss/n_batches:.4f} Acc={running_acc/n_batches:.4f}")

            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    probs = torch.sigmoid(model(imgs))
                    preds = (probs > fixed_thresh).float()
                    lbls = labels.float().unsqueeze(1)
                    val_correct += (preds == lbls).float().sum().item()
                    val_total += preds.numel()
            val_acc = val_correct / val_total

            print(f"[Clf] Val Acc={val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), cfg.get("classification.save_path"))

                print(f"[Classifier] Saved best Acc: {best_val_acc:.4f}")
        
            if logger:
                logger.log_scalar("Train/Clf_Accuracy", running_acc / n_batches, step=epoch_counter)
                logger.log_scalar("Train/Clf_Loss", running_loss / n_batches, step=epoch_counter)
                logger.log_scalar("Eval/Clf_BestAccuracy", best_val_acc, step=epoch_counter)
                logger.log_scalar("Eval/Clf_ValAccuracy", val_acc, step=epoch_counter)

    model.load_state_dict(torch.load(cfg.get("classification.save_path"), map_location=device))
    print(f"[Classifier] Restored best model @ Acc={best_val_acc:.4f}")

    return model