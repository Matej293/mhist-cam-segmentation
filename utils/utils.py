import numpy as np
import torch.nn as nn
import torch
import cv2
import torchstain
import albumentations as A

# Focal Tversky loss function for Classification
class ClassifierFocalTverskyLoss(nn.Module):
    """
    Focal Tversky loss for imbalanced binary classification.
    """
    def __init__(self, alpha=0.7, beta=0.3, gamma=4./3., smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        tp = (probs * targets).sum()
        fn = ((1 - probs) * targets).sum()
        fp = (probs * (1 - targets)).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return (1 - tversky) ** self.gamma

# Torchstain: Macenko Normalizer
class TorchstainNormalize(A.ImageOnlyTransform):
    """Albumentations wrapper around torchstain.MacenkoNormalizer."""
    def __init__(self, always_apply=False, p=0.7):
        super().__init__(always_apply, p)
        macenko = torchstain.normalizers.MacenkoNormalizer()
        self.normalizer = macenko

    def apply(self, img, **kwargs):
        # img: H×W×C uint8 RGB
        img_f = img.astype("float32") / 255.0
        out = self.normalizer.normalize(img_f)
        out = (np.clip(out, 0, 1) * 255).astype("uint8")
        return out

    def get_transform_init_args_names(self):
        return ()
    

# Remove small connected components in binary masks
def remove_small_regions_batch(
    preds: torch.Tensor,
    tissue: torch.Tensor,
    min_size: int
) -> torch.Tensor:
    """
    preds:   (N,1,H,W)  binary torch tensor (0/1) of predicted mask
    tissue:  (N,1,H,W)  binary torch tensor mask of valid tissue
    min_size: int       minimum connected-component size (in pixels) to keep
    
    Returns a tensor of shape (N,1,H,W) where any connected component
    that lies within 'tissue' and has area < min_size is removed (set to 0).
    """
    # Ensure both inputs are 4D
    if preds.dim() != 4 or tissue.dim() != 4:
        raise ValueError(f"remove_small_regions_batch expects 4D tensors, got preds.dim()={preds.dim()}, tissue.dim()={tissue.dim()}")

    device = preds.device
    dtype = preds.dtype
    out = preds.clone()

    N, C, H, W = preds.shape
    if C != 1:
        raise ValueError(f"remove_small_regions_batch expects preds.shape[1] == 1, got {C}")

    for i in range(N):
        p_np = (preds[i, 0].cpu().numpy() > 0).astype(np.uint8)
        t_np = (tissue[i, 0].cpu().numpy() > 0).astype(np.uint8)

        # Only pixels inside tissue
        mask = p_np * t_np  # shape (H, W), 0/1

        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        clean = np.zeros_like(mask, dtype=np.uint8)

        # Keep only components whose area >= min_size
        for lab in range(1, num_labels):
            area = stats[lab, cv2.CC_STAT_AREA]
            if area >= min_size:
                clean[labels == lab] = 1

        # Write back into out tensor, preserving dtype and device
        out[i, 0] = (
            torch.from_numpy(clean)
            .to(device)
            .to(dtype)
        )

    return out
