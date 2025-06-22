import os 
import wandb
import numpy as np
import matplotlib.pyplot as plt


class WandbLogger():
    """Minimal Weights & Biases logger with figure logging support."""
    def __init__(self, project=None, name=None, config=None, reuse=False):
        super().__init__()
        self.train_epoch_step = 1
        self.eval_step = 1

        # Set wandb directory to a subfolder in the current working directory
        wandb_dir = os.path.join(os.getcwd(), "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        os.environ["WANDB_DIR"] = wandb_dir

        # Only initialize a new run if not reusing and no run exists
        if not reuse and (getattr(wandb, "run", None) is None or wandb.run is False):
            wandb.init(project=project, name=name, config=config, dir=wandb_dir)

        self._define_metric_groups()
            
    def _define_metric_groups(self):
        """Define metric groups with their own step counters."""
        wandb.define_metric("train_epoch_step", summary="max")
        wandb.define_metric("eval_step", summary="max")

        wandb.define_metric("Train/Clf_Accuracy", step_metric="train_epoch_step")
        wandb.define_metric("Train/Clf_Loss", step_metric="train_epoch_step")
        wandb.define_metric("Train/SegmLoss", step_metric="train_epoch_step")
        wandb.define_metric("Train/Segm_p_hp", step_metric="train_epoch_step")

        wandb.define_metric("Eval/*", step_metric="eval_step")

    def _update_step_counter(self, tag, log_dict, step=None):
        """Update step counter and attach the correct step metric for logging."""
        if tag.startswith("Eval/"):
            if step is not None:
                self.eval_step = step
            else:
                self.eval_step += 1
            log_dict["eval_step"] = self.eval_step
        else:
            if step is not None:
                self.train_epoch_step = step
            else:
                self.train_epoch_step += 1
            log_dict["train_epoch_step"] = self.train_epoch_step
        return log_dict
    
    def _log_figure(self, tag, figure, step=None):
        """Log a matplotlib figure to wandb."""
        if figure is None:
            return
        try:
            img = wandb.Image(figure)
            wandb.log(self._update_step_counter(tag, {tag: img}, step))
        except Exception as e:
            print(f"Warning: Failed to log figure for tag '{tag}': {e}")
        finally:
            plt.close(figure)


    def log_scalar(self, tag, value, step=None):
        """Log a scalar value to wandb."""
        if value is None:
            return
        try:
            value = float(value)
        except (TypeError, ValueError):
            print(f"Warning: Unable to convert value '{value}' for tag '{tag}' to float.")
            return
        wandb.log(self._update_step_counter(tag, {tag: value}, step))
    

    def log_images(self, tag, images, step=None, max_images=4):
        """Log a batch of images to wandb."""
        if images is None or len(images) == 0:
            return

        imgs = []
        for img in images[:max_images]:
            arr = img.detach().cpu().numpy() if hasattr(img, "detach") else (
                img.cpu().numpy() if hasattr(img, "cpu") else np.array(img)
            )
            # Handle channel-first (C, H, W) to channel-last (H, W, C)
            if arr.ndim == 3 and arr.shape[0] in [1, 3]:
                arr = np.transpose(arr, (1, 2, 0))
            # Normalize if needed
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 1)
            imgs.append(wandb.Image(arr))
        if imgs:
            wandb.log(self._update_step_counter(tag, {tag: imgs}, step))


    def close(self):
        """Safely finish the wandb run if active."""
        run = getattr(wandb, "run", None)
        if run is not None and run is not False:
            try:
                wandb.finish()
            except Exception as e:
                print(f"Warning: wandb.finish() failed: {e}")
