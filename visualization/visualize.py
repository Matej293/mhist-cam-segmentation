import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

# import custom modules
from utils.utils import remove_small_regions_batch
from concurrent.futures import ThreadPoolExecutor

def visualize_segmentation_results(images, masks, predictions, probabilities, step=None, logger=None, max_samples=6, export_detailed: bool = False, export_path: str = None, batch_idx: int = 0):
    """
    Create comprehensive visualization of segmentation results.
    If export_detailed is True, additionally save the _detailed_viz plots to disk under 'config.logging.visualization_path'.
    """
    # Supress matplotlib warnings about too many open figures
    plt.rc('figure', max_open_warning = 0)

    if logger is None or images is None or len(images) == 0:
        return

    if export_detailed:
        export_path = export_path or logger.get('logging.visualization_path', './data/visualizations')
        os.makedirs(export_path, exist_ok=True)

    batch_size = min(len(images), max_samples)
    # Log only the first batch_idx=0 to avoid cluttering WandB
    log_to_wandb = (batch_idx == 0)

    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(images.device)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(images.device)

    images_display = images.clone()
    images_display = images_display * std + mean
    images_display = images_display.clamp(0,1)

    predictions = remove_small_regions_batch(predictions, masks, min_size=500)

    _basic_viz(images_display, masks, predictions, step, logger, batch_size, log_to_wandb)
    _detailed_viz(images_display, masks, predictions, probabilities, step, logger, batch_size, log_to_wandb, export_detailed=export_detailed, export_path=export_path, batch_idx=batch_idx)
    _error_viz(images_display, masks, predictions, step, logger, batch_size, log_to_wandb)
    _uncertainty_viz(images_display, probabilities, step, logger, batch_size, log_to_wandb)

    plt.close('all')


def _basic_viz(images, masks, predictions, step, logger, batch_size, log_to_wandb):
    """Log input images, ground-truth masks, and predicted masks, with SSA=red, HP=blue."""
    if not log_to_wandb:
        return
    # Input images
    logger.log_images("Viz/Input", images, step=step, max_images=batch_size)

    # Color‐coded GT and prediction
    gt_colored, pr_colored = [], []
    for i in range(batch_size):
        m = masks[i][0].cpu().numpy()       # binary 1=SSA,0=HP
        p = predictions[i][0].cpu().numpy() # same

        h,w = m.shape
        cm_gt = np.zeros((3,h,w), dtype=np.float32)
        cm_pr = np.zeros((3,h,w), dtype=np.float32)

        # SSA pixels in Red channel, HP pixels in Blue channel
        cm_gt[0] = m        # red = m
        cm_gt[2] = 1.0 - m  # blue = inverse

        cm_pr[0] = p
        cm_pr[2] = 1.0 - p

        gt_colored.append(torch.from_numpy(cm_gt))
        pr_colored.append(torch.from_numpy(cm_pr))

    gt_tensor = torch.stack(gt_colored)
    pr_tensor = torch.stack(pr_colored)

    logger.log_images("Viz/GroundTruth", gt_tensor, step=step, max_images=batch_size)
    logger.log_images("Viz/Prediction",  pr_tensor, step=step, max_images=batch_size)


def _detailed_viz(images, masks, predictions, probabilities, step, logger, batch_size, log_to_wandb, export_detailed: bool = False, export_path: str = None, batch_idx: int = 0):
    """Side-by-side original / GT / pred / heatmap / overlays."""
    try:
        # Set n_export to one batch (max_samples) if not exporting images to disk, else to the number of images
        n_export = images.shape[0] if export_detailed else batch_size

        # If exporting, use a non-interactive backend and avoid showing figures
        imgs_np = images.permute(0,2,3,1).cpu().numpy()
        masks_np = masks[:,0].cpu().numpy()
        preds_np = predictions[:,0].cpu().numpy()
        probs_np = probabilities[:,0].cpu().numpy()

        def save_fig(fig, fname):
            fig.savefig(fname, bbox_inches='tight', dpi=150)
            plt.close(fig)

        save_tasks = []
        # If exporting, use a thread pool for parallel saving
        executor = ThreadPoolExecutor(max_workers=4) if export_detailed and export_path else None

        for i in range(n_export):
            fig = plt.figure(figsize=(15,10))
            gs  = gridspec.GridSpec(2,3, figure=fig)

            img  = imgs_np[i]
            m    = masks_np[i]
            p    = preds_np[i]
            prob = probs_np[i]

            # Original
            ax = fig.add_subplot(gs[0,0]); ax.imshow(img); ax.axis('off'); ax.set_title("Original")

            # GT
            ax = fig.add_subplot(gs[0,1])
            h, w = m.shape
            gt_display = np.zeros((h, w, 3), dtype=np.float32)
            gt_display[:, :, 0] = m           # red channel = SSA
            gt_display[:, :, 2] = 1.0 - m     # blue channel = HP/background
            ax.set_title("Ground Truth: SSA" if m.sum()>0 else "Ground Truth: HP")
            ax.imshow(gt_display); ax.axis('off')

            # Pred
            ax = fig.add_subplot(gs[0,2])
            h, w = p.shape
            pred_display = np.zeros((h, w, 3), dtype=np.float32)
            pred_display[:, :, 0] = p           # red = predicted SSA
            pred_display[:, :, 2] = 1.0 - p     # blue = predicted HP/background
            ax.set_title("Prediction: SSA" if p.sum()>0 else "Prediction: HP")
            ax.imshow(pred_display); ax.axis('off')

            # Probability heatmap
            ax = fig.add_subplot(gs[1,0])
            im = ax.imshow(prob, cmap='jet', vmin=0, vmax=1)
            ax.set_title("Prob Map (red=SSA)")
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Overlay GT
            ax = fig.add_subplot(gs[1,1]); ax.imshow(img)
            ov = np.zeros_like(img)
            if m.sum() > 0:    ov[:,:,0]=m*0.7  # red overlay
            else:              ov[:,:,2]=(1-m)*0.7
            ax.imshow(ov, alpha=0.5); ax.axis('off'); ax.set_title("GT Overlay")

            # Overlay Pred
            ax = fig.add_subplot(gs[1,2]); ax.imshow(img)
            ov = np.zeros_like(img)
            if p.sum() > 0:    ov[:,:,0]=p*0.7
            else:              ov[:,:,2]=(1-p)*0.7
            ax.imshow(ov, alpha=0.5); ax.axis('off'); ax.set_title("Pred Overlay")

            plt.tight_layout()

            # Log only the first max_samples figures to WandB
            if i < batch_size and log_to_wandb:
                logger._log_figure(f"Viz/Detailed_{i+1}", fig, step=step)

            if export_detailed and export_path:
                fname = os.path.join(export_path, f"detailed_epoch{step or 0:03d}_b{batch_idx:03d}_i{i+1:03d}.png")
                if executor:
                    # Save in parallel
                    save_tasks.append(executor.submit(save_fig, fig, fname))
                else:
                    fig.savefig(fname, bbox_inches='tight', dpi=150)
                    plt.close(fig)
            else:
                plt.close(fig)

        if executor:
            # Wait for all saves to finish
            for t in save_tasks:
                t.result()
            executor.shutdown()

    except Exception as e:
        plt.close('all')
        print(f"Warning in detailed_viz: {e}")


def _error_viz(images, masks, predictions, step, logger, batch_size, log_to_wandb):
    """Error map: TP/FP/FN/TN and legend."""
    if not log_to_wandb:
        return
    try:
        # detailed one‐by‐one
        for i in range(batch_size):
            img    = images[i].permute(1,2,0).cpu().numpy()
            m_np   = masks[i][0].cpu().numpy().astype(bool)
            p_np   = predictions[i][0].cpu().numpy().astype(bool)

            err = np.zeros((*m_np.shape,3), dtype=np.float32)
            err[np.logical_and(p_np, m_np)]               = [1,1,1]    # TP white
            err[np.logical_and(p_np, np.logical_not(m_np))] = [1,0,0]  # FP red
            err[np.logical_and(np.logical_not(p_np), m_np)] = [0,0,1]  # FN blue
            # TN are left black

            fig = plt.figure(figsize=(12,8))
            gs  = gridspec.GridSpec(2,2, figure=fig, height_ratios=[3,1])

            ax = fig.add_subplot(gs[0,0]); ax.imshow(img); ax.axis('off'); ax.set_title("Orig")
            ax = fig.add_subplot(gs[0,1]); ax.imshow(err); ax.axis('off'); ax.set_title("Error Map")

            # Legend
            ax = fig.add_subplot(gs[1,:]); ax.axis('off')
            legends = [
                mpatches.Patch(color='white', label='TP (correct SSA)'),
                mpatches.Patch(color='red',   label='FP (called SSA, was HP)'),
                mpatches.Patch(color='blue',  label='FN (missed SSA)'),
                mpatches.Patch(color='black', label='TN (correct HP)')
            ]
            ax.legend(handles=legends, loc='center', frameon=True, title="Errors")

            plt.tight_layout()
            logger._log_figure(f"Viz/Error_{i+1}", fig, step=step)
            plt.close(fig)

        # grid of tiny error maps
        emaps = []
        for i in range(batch_size):
            m_np = masks[i][0].cpu().numpy().astype(bool)
            p_np = predictions[i][0].cpu().numpy().astype(bool)
            err = np.zeros((*m_np.shape,3), dtype=np.float32)
            err[np.logical_and(p_np, m_np)]               = [1,1,1]
            err[np.logical_and(p_np, np.logical_not(m_np))] = [1,0,0]
            err[np.logical_and(np.logical_not(p_np), m_np)] = [0,0,1]
            emaps.append(torch.from_numpy(err.transpose(2,0,1)))

        emaps = torch.stack(emaps)
        logger.log_images("Viz/ErrorGrid", emaps, step=step, max_images=batch_size)

    except Exception as e:
        plt.close('all')
        print(f"Warning in error_viz: {e}")


def _uncertainty_viz(images, probabilities, step, logger, batch_size, log_to_wandb):
    """Uncertainty = 1 - 2|p-0.5|, plus explanation."""
    if not log_to_wandb:
        return
    try:
        ums = []
        for i in range(batch_size):
            prob = probabilities[i][0].cpu().numpy()
            un   = 1.0 - 2.0 * np.abs(prob - 0.5)
            cmap = plt.cm.get_cmap('viridis')
            col  = cmap(un)[:,:,:3]
            ums.append(torch.from_numpy(col.transpose(2,0,1)))

        ums = torch.stack(ums)
        logger.log_images("Viz/UncertaintyMap", ums, step=step, max_images=batch_size)

        # explanatory figure
        fig = plt.figure(figsize=(15,8))
        gs  = gridspec.GridSpec(2,3, figure=fig, height_ratios=[3,1])
        for i in range(min(3,batch_size)):
            ax = fig.add_subplot(gs[0,i])
            prob = probabilities[i][0].cpu().numpy()
            un   = 1.0 - 2.0 * np.abs(prob - 0.5)
            im   = ax.imshow(un, cmap='viridis'); ax.axis('off')
            ax.set_title(f"Uncertainty {i+1}")
        cax = fig.add_axes([0.92,0.5,0.02,0.35])
        fig.colorbar(im, cax=cax).set_label('Uncertainty')
        ax = fig.add_subplot(gs[1,:]); ax.axis('off')
        txt = (
            "Uncertainty map: yellow=high (p≈0.5),\n"
            "blue/purple=low (p≈0 or 1)."
        )
        ax.text(0.5,0.5,txt,ha='center',va='center',fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5",fc="white",alpha=0.9))
        logger._log_figure("Viz/UncertaintyExp", fig, step=step)
        plt.close(fig)

    except Exception as e:
        plt.close('all')
        print(f"Warning in uncertainty_viz: {e}")
