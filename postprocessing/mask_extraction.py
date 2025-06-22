import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
from torchvision import transforms
from torchcam.methods import SmoothGradCAMpp
from skimage.morphology import remove_small_objects
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
from concurrent.futures import ThreadPoolExecutor, as_completed

# import custom modules
from datasets.mhist import MHISTDataset
from config.config_manager import ConfigManager

def extract_pseudo_masks(
        model: nn.Module,
        dataset: MHISTDataset,
        cfg: ConfigManager,
        device: torch.device):
    """
    Generate CAM-based pseudo-masks for SSA images, apply morphology + CRF, save to data.mask_dir.
    """
    model.eval()
    cam_extractor = SmoothGradCAMpp(model, target_layer="layer4")

    cam_size = tuple(cfg.get("postprocessing.cam_resize", [224, 224]))
    mean = cfg.get("postprocessing.cam_normalize_mean", [0.485, 0.456, 0.406])
    std = cfg.get("postprocessing.cam_normalize_std", [0.229, 0.224, 0.225])
    cam_tf = transforms.Compose([
        transforms.Resize(cam_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    out_dir = cfg.get("data.mask_dir")
    os.makedirs(out_dir, exist_ok=True)

    thresh_q     = cfg.get("postprocessing.threshold_quantile", 0.25)
    open_k       = cfg.get("postprocessing.open_kernel", 3)
    close_k      = cfg.get("postprocessing.close_kernel", 5)
    min_obj      = cfg.get("postprocessing.min_object_size", 64)
    crf_iters    = cfg.get("postprocessing.crf_iters", 3)
    gauss_sxy    = cfg.get("postprocessing.gaussian_sxy", 3)
    gauss_compat = cfg.get("postprocessing.gaussian_compat", 3)
    bilat_sxy    = cfg.get("postprocessing.bilateral_sxy", 80)
    bilat_srgb   = cfg.get("postprocessing.bilateral_srgb", 13)
    bilat_compat = cfg.get("postprocessing.bilateral_compat", 10)

    print("[CAM] Generating pseudo-masks...")

    executor = ThreadPoolExecutor(max_workers=4)
    futures = []

    for idx in tqdm(range(len(dataset)), desc="CAM Extract", ncols=80):
        fname = dataset.data.iloc[idx]["Image Name"]
        pil = Image.open(os.path.join(cfg.get("data.img_dir"), fname)).convert("RGB")
        W, H = pil.size
        orig = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        _, bg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        tissue_mask = (bg == 0).astype(np.uint8)

        _, label = dataset[idx]  # 0=HP, 1=SSA

        inp = cam_tf(pil).unsqueeze(0).to(device)
        model.zero_grad()
        logits = model(inp)

        if label == 1:
            cam = cam_extractor(0, logits)[0].squeeze().cpu().numpy()
            cam = cv2.resize(cam, (W, H), interpolation=cv2.INTER_CUBIC)
            cam = (cam - cam.min()) / (cam.ptp() + 1e-8)
        else:
            cam = np.zeros((H, W), dtype=np.float32)

        cam *= tissue_mask
        q = np.quantile(cam[tissue_mask == 1], thresh_q) if label == 1 else 1.0
        mask = ((cam > q) & (tissue_mask == 1)).astype(np.uint8)

        ko = np.ones((open_k, open_k), np.uint8)
        kc = np.ones((close_k, close_k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ko)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc)
        mask = remove_small_objects(mask.astype(bool), min_size=min_obj).astype(np.uint8)

        prob = np.stack([1 - mask, mask], axis=0)
        U = unary_from_softmax(prob.reshape(2, -1))
        if U.shape != (2, H * W):
            raise ValueError(f"Unary shape {U.shape}, expected (2, {H*W})")

        dcrf_model = dcrf.DenseCRF2D(W, H, 2)
        dcrf_model.setUnaryEnergy(U)
        dcrf_model.addPairwiseGaussian(sxy=gauss_sxy, compat=gauss_compat)
        dcrf_model.addPairwiseBilateral(sxy=bilat_sxy, srgb=bilat_srgb, rgbim=orig, compat=bilat_compat)
        Q = np.array(dcrf_model.inference(crf_iters))
        refined = Q.argmax(axis=0).reshape(H, W).astype(np.uint8)

        out_name = os.path.splitext(fname)[0] + "_mask.png"
        out_path = os.path.join(out_dir, out_name)
        futures.append(
            executor.submit(
                cv2.imwrite,
                out_path,
                (refined * 255)
            )
        )

    # Wait for all futures to complete
    for f in as_completed(futures):
        _ = f.result()   # True if write succeeded
    
    executor.shutdown()
    print("[CAM] Pseudo-masks done.")