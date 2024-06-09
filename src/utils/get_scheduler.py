import torch

def get_scheduler(optimizer, config: dict) -> torch.optim.lr_scheduler._LRScheduler:
    """Gets the learning rate scheduler based on the configuration.

    Args:
        optimizer: The PyTorch optimizer for which to create the scheduler.
        config: A dictionary containing the configuration parameters.

    Returns:
        The initialized learning rate scheduler.

    Raises:
        ValueError: If an unsupported scheduler type is specified in the config.
    """
    scheduler_type = config["lr_scheduler"]
    num_epochs = config["num_epochs"]

    if scheduler_type == "linear":
        step_size = config.get("step_size", 5)  # Default step size is 5 epochs
        gamma = config.get("gamma", 0.1)  # Default gamma is 0.1
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == "cosine":
        T_max = num_epochs  # Cosine annealing for the entire training duration
        eta_min = 0.0  # Minimum learning rate (optional)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    return scheduler
