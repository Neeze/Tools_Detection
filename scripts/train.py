import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.faster_rcnn import fasterrcnn_resnet18
from src.utils.dataloader import create_dataloader
from src.utils.get_optimizer import get_optimizer
from src.utils.get_scheduler import get_scheduler
from src.utils.logger import TrainingLogger
from src.utils.general_utils import read_config
from src.utils.engine import train_one_epoch, evaluate, save_model


def main():
    config = read_config("config/faster_rcnn18_config.yaml")  # Load config
    logger = TrainingLogger(config)  # Initialize logger (e.g., Weights & Biases)

    device = torch.device('cuda' if torch.cuda.is_available() and config["device"] == 'gpu' else 'cpu')
    print(f"Using device: {device}")

    # Create data loaders
    train_loader = create_dataloader(config, is_train=True)
    val_loader = create_dataloader(config, is_train=False)

    # Create model
    model = fasterrcnn_resnet18(num_classes=config["num_classes"], pretrained=True, coco_model=True).to(device)
    
    # Create optimizer and learning rate scheduler
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
   
    # Training loop
    start_epoch = 0
    num_epochs = config["num_epochs"]
    best_val_loss = float("inf")

    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        # Train for one epoch
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, logger)
        scheduler.step() # Update learning rate

        # Evaluate on the validation set 
        if (epoch + 1) % config.get("val_interval", 1) == 0: # Validate every epoch by default
            val_loss = evaluate(model, val_loader, device, logger)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model(model, epoch, optimizer, scheduler, config)

if __name__ == "__main__":
    main()
