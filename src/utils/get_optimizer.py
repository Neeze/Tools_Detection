import torch

def get_optimizer(model, config: dict) -> torch.optim.Optimizer:
    """Gets the optimizer based on the configuration.

    Args:
        model: The PyTorch model to optimize.
        config: A dictionary containing the configuration parameters.

    Returns:
        The initialized optimizer.

    Raises:
        ValueError: If an unsupported optimizer type is specified in the config.
    """

    optimizer_type = config["optimizer"]
    learning_rate = config["learning_rate"]
    weight_decay = config.get("weight_decay", 0.0)  # Default weight decay is 0.0 if not specified

    if optimizer_type == "SGD":
        momentum = config.get("momentum", 0.9)  # Default momentum is 0.9
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=learning_rate, 
                                    momentum=momentum, 
                                    weight_decay=weight_decay)
    elif optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), 
                                     lr=learning_rate, 
                                     weight_decay=weight_decay)
    elif optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), 
                                      lr=learning_rate, 
                                      weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    return optimizer
