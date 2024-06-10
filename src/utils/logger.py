import wandb
import torch

class TrainingLogger:
    def __init__(self, config: dict):
        """Initializes the Weights & Biases logger with configuration.

        Args:
            config: A dictionary containing the configuration parameters.
        """
        wandb.init(project=config["project_name"], config=config)
        self.config = config

    def log_epoch_metrics(self, epoch: int, train_loss: float, val_loss: float, lr: float, **kwargs):
        """Logs metrics for a training epoch.

        Args:
            epoch: The current epoch number.
            train_loss: The average training loss for the epoch.
            val_loss: The average validation loss for the epoch.
            lr: The learning rate used during the epoch.
            **kwargs: Additional metrics to log (e.g., mAP, accuracy).
        """
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr,
            **kwargs
        }

        # Log accelerator (GPU/CPU) and GPU metrics if available
        if torch.cuda.is_available():
            metrics["accelerator"] = "gpu"
            metrics["gpu_mem_alloc"] = torch.cuda.memory_allocated() / 1024**2  # In MB
            metrics["gpu_mem_reserved"] = torch.cuda.memory_reserved() / 1024**2  # In MB
        else:
            metrics["accelerator"] = "cpu"

        wandb.log(metrics)

    def watch_model(self, model):
        """Watches the PyTorch model to log gradients and parameters.

        Args:
            model: The PyTorch model to watch.
        """
        wandb.watch(model, log="all")
