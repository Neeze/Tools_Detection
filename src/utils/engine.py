import torch
import time
import os
from tqdm import tqdm
from .logger import *
from .metrics import calculate_map
import torchvision
from typing import Tuple, List, Dict, Optional
import torch
from torch import Tensor
from collections import OrderedDict
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.rpn import concat_box_prediction_layers

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


def eval_forward(model, images, targets):
    """
    Args:
        images (list[Tensor]): images to be processed
        targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)
    Returns:
        result (list[BoxList] or dict[Tensor]): the output from the model.
            It returns list[BoxList] contains additional fields
            like `scores`, `labels` and `mask` (for Mask R-CNN models).
    """
    model.eval()

    original_image_sizes: List[Tuple[int, int]] = []
    for img in images:
        val = img.shape[-2:]
        assert len(val) == 2
        original_image_sizes.append((val[0], val[1]))

    images, targets = model.transform(images, targets)

    # Check for degenerate boxes
    # TODO: Move this to a function
    if targets is not None:
        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # print the first degenerate box
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                raise ValueError(
                    "All bounding boxes should have positive height and width."
                    f" Found invalid box {degen_bb} for target at index {target_idx}."
                )

    features = model.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])
    model.rpn.training=True
    #model.roi_heads.training=True


    #####proposals, proposal_losses = model.rpn(images, features, targets)
    features_rpn = list(features.values())
    objectness, pred_bbox_deltas = model.rpn.head(features_rpn)
    anchors = model.rpn.anchor_generator(images, features_rpn)

    num_images = len(anchors)
    num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
    num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
    objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
    # apply pred_bbox_deltas to anchors to obtain the decoded proposals
    # note that we detach the deltas because Faster R-CNN do not backprop through
    # the proposals
    proposals = model.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
    proposals = proposals.view(num_images, -1, 4)
    proposals, scores = model.rpn.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

    proposal_losses = {}
    assert targets is not None
    labels, matched_gt_boxes = model.rpn.assign_targets_to_anchors(anchors, targets)
    regression_targets = model.rpn.box_coder.encode(matched_gt_boxes, anchors)
    loss_objectness, loss_rpn_box_reg = model.rpn.compute_loss(
        objectness, pred_bbox_deltas, labels, regression_targets
    )
    proposal_losses = {
        "loss_objectness": loss_objectness,
        "loss_rpn_box_reg": loss_rpn_box_reg,
    }

    #####detections, detector_losses = model.roi_heads(features, proposals, images.image_sizes, targets)
    image_shapes = images.image_sizes
    proposals, matched_idxs, labels, regression_targets = model.roi_heads.select_training_samples(proposals, targets)
    box_features = model.roi_heads.box_roi_pool(features, proposals, image_shapes)
    box_features = model.roi_heads.box_head(box_features)
    class_logits, box_regression = model.roi_heads.box_predictor(box_features)

    result: List[Dict[str, torch.Tensor]] = []
    detector_losses = {}
    loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
    detector_losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
    boxes, scores, labels = model.roi_heads.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
    num_images = len(boxes)
    for i in range(num_images):
        result.append(
            {
                "boxes": boxes[i],
                "labels": labels[i],
                "scores": scores[i],
            }
        )
    detections = result
    detections = model.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]
    model.rpn.training=False
    model.roi_heads.training=False
    losses = {}
    losses.update(detector_losses)
    losses.update(proposal_losses)
    return losses, detections


@torch.no_grad() 
def evaluate(model, data_loader, device, logger, score_threshold=0.05):  # Add score_threshold
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

        loss_dict, detections = eval_forward(model, images, targets) # Only for loss calculation
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        epoch_loss += loss_value
    # Calculate mAP (you'll need to adapt this to your specific mAP implementation)
    # mAP = calculate_map(all_targets, all_predictions, logger.config["num_classes"])
    # TODO: Implement mAP calculation for classes
    # Log validation loss and mAP
    epoch_loss /= len(data_loader)
    logger.log_epoch_metrics(None, None, epoch_loss, None, mAP=None)

    return epoch_loss



def save_model(model, epoch, optimizer, scheduler, config):
    """Saves the model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    
    os.mkdir("checkpoints") if not os.path.exists("checkpoints") else None
    torch.save(checkpoint, f"checkpoints/{config['backbone']}_epoch_{epoch}.pth")
