#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py

Trains a CAM-based weakly supervised classifier + DeepLabV3+ segmentation
on the MHIST dataset, generates pseudo-masks, and evaluates performance.
"""

import torch
import random
import numpy as np
from tqdm import tqdm

# import custom modules
from config.config_manager import ConfigManager
from metrics.wandb_logger import WandbLogger
from datasets.mhist import get_mhist_loaders
from models.classifier import train_classifier
from models.segmentation import train_segmentation
from postprocessing.mask_extraction import extract_pseudo_masks
from visualization.visualize import visualize_segmentation_results


def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg = ConfigManager("config/default_config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = WandbLogger(
        project=cfg.get("logging.project"),
        name=cfg.get("logging.run_name"),
        config=cfg.config,
    )

    # Classification loaders
    # clf_train_loader, clf_val_loader = get_mhist_loaders(cfg , task="classification")

    # Train classifier
    # clf_model = train_classifier(clf_train_loader, clf_val_loader, cfg, device, logger)
    
    # Generate pseudo-masks
    # if cfg.get("postprocessing.enabled"):
        # Build a dataset for CAM extraction
        # cam_dataset, _ = get_mhist_loaders(cfg, task="all")
        # Extract pseudo-masks using the trained classifier
        # extract_pseudo_masks(clf_model, cam_dataset, cfg, device)

    # Segmentation loaders
    ssa_loader, hp_loader, val_loader = get_mhist_loaders(cfg, task="segmentation")

    # Segmentation model
    seg_model = train_segmentation(ssa_loader, hp_loader, val_loader, cfg, device, logger)

    # Final evaluation and visualization
    print("[Main] Final eval & visualization...")
    
    # Set this to True if you want to export visualizations to disk
    export_detailed = False
    preview_images = 6  # how many you want to log if not exporting full disk

    # set the total number of images to visualize
    total = len(val_loader.dataset) if export_detailed else preview_images

    with torch.no_grad():
        pbar = tqdm(total=total, desc="Visualization", ncols=80, unit="img")
        images_done = 0

        for batch_idx, (imgs, msks, _) in enumerate(val_loader):
            bs = imgs.size(0)

            imgs, msks = imgs.to(device), msks.to(device)
            outs = seg_model(imgs)
            probs = torch.sigmoid(outs)
            preds = (probs > cfg.get("training.threshold")).float()

            visualize_segmentation_results(
                images=imgs,
                masks=msks,
                predictions=preds,
                probabilities=probs,
                step=cfg.get("training.epochs"),
                logger=logger,
                max_samples=preview_images,         # Limit to n samples for Wandb logging
                export_detailed=export_detailed,    # Save detailed visualizations under 'config.logging.visualization_path'
                export_path=cfg.get("logging.visualization_path", "./data/visualizations"),
                batch_idx=batch_idx
            )

            # update progress bar
            to_update = bs if export_detailed else min(bs, preview_images - images_done)
            pbar.update(to_update)
            images_done += to_update

            # stop if we’ve done our preview
            if not export_detailed and images_done >= preview_images:
                break

        pbar.close()


if __name__ == "__main__":
    main()