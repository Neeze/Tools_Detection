import torch
import time
import os
from tqdm import tqdm
import logger
from src.utils.metrics import calculate_map
import torchvision

def train_one_epoch(model, optimizer, data_loader, device, epoch, logger):
    model.train()
    epoch_loss = 0.0
    if device.type == "cuda":
        model.to(device)
    for batch_idx, (images, targets) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch + 1}/{logger.config['num_epochs']}")):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        epoch_loss += loss_value

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Log batch-level metrics (every N steps or at the end of the epoch)
        if (batch_idx + 1) % logger.config.get("log_interval", 100) == 0 or batch_idx == len(data_loader) - 1:
            logger.log_epoch_metrics(
                epoch, 
                loss_value, # Use loss_value for current batch loss
                None, 
                optimizer.param_groups[0]["lr"], 
                batch_idx=batch_idx
            )

    # Log epoch-level metrics (average loss)
    epoch_loss /= len(data_loader)
    logger.log_epoch_metrics(epoch, epoch_loss, None, optimizer.param_groups[0]["lr"]) 
    
    return epoch_loss


@torch.no_grad() 
def evaluate(model, data_loader, device, score_threshold=0.05):  # Add score_threshold
    model.eval()
    epoch_loss = 0.0
    all_targets = []
    all_predictions = []

    for images, targets in tqdm(data_loader, desc="Evaluating"):
        images = list(image.to(device) for image in images)

        # Ensure targets are on the correct device
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets] 

        # Forward pass
        outputs = model(images)  # Get model predictions
        
        # Post-process predictions (non-max suppression)
        keep_idx = torchvision.ops.nms(
            outputs[0]["boxes"], 
            outputs[0]["scores"], 
            iou_threshold=0.6  # Adjust IoU threshold if needed
        )
        outputs[0]["boxes"] = outputs[0]["boxes"][keep_idx]
        outputs[0]["labels"] = outputs[0]["labels"][keep_idx]
        outputs[0]["scores"] = outputs[0]["scores"][keep_idx]

        # Filter predictions by score threshold
        for i, output in enumerate(outputs):
            keep_idx = output["scores"] >= score_threshold
            all_predictions.append(
                {
                    "boxes": output["boxes"][keep_idx].cpu().numpy(),
                    "labels": output["labels"][keep_idx].cpu().numpy(),
                    "scores": output["scores"][keep_idx].cpu().numpy(),
                }
            )
            all_targets.append(targets[i]) 

        # Calculate and accumulate loss (if needed for evaluation)
        loss_dict = model(images, targets)  # Only for loss calculation
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        epoch_loss += loss_value

    # Calculate mAP (you'll need to adapt this to your specific mAP implementation)
    mAP = calculate_map(all_targets, all_predictions, model.num_classes)

    # Log validation loss and mAP
    epoch_loss /= len(data_loader)
    logger.log_epoch_metrics(None, None, epoch_loss, None, mAP=mAP)

    return epoch_loss, mAP



def save_model(model, epoch, optimizer, scheduler, config):
    """Saves the model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    
    os.mkdir("checkpoints") if not os.path.exists("checkpoints") else None
    torch.save(checkpoint, f"checkpoints/{config['model_name']}_epoch_{epoch}.pth")
