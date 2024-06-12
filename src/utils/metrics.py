import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics import Accuracy


def calculate_map(targets, predictions, num_classes):
    """Calculates mean Average Precision (mAP) for object detection.

    Args:
        targets: List of dictionaries, each containing ground truth 'boxes' and 'labels'.
        predictions: List of dictionaries, each containing predicted 'boxes', 'labels', and 'scores'.
        num_classes: The number of object classes (including background).

    Returns:
        The calculated mAP value.
    """
    
    metric = MeanAveragePrecision(num_classes=num_classes)
    metric.update(preds=predictions, target=targets)
    return metric.compute()["map"]


def calculate_accuracy(targets, predictions, iou_threshold=0.5):
    """Calculates accuracy for object detection based on IoU threshold.

    Args:
        targets: List of dictionaries, each containing ground truth 'boxes' and 'labels'.
        predictions: List of dictionaries, each containing predicted 'boxes', 'labels', and 'scores'.
        iou_threshold: The IoU threshold to consider a prediction as correct.

    Returns:
        The calculated accuracy value.
    """
    total_correct = 0
    total_samples = 0

    for target, prediction in zip(targets, predictions):
        target_boxes = target["boxes"]
        target_labels = target["labels"]
        pred_boxes = prediction["boxes"]
        pred_labels = prediction["labels"]

        for pred_box, pred_label in zip(pred_boxes, pred_labels):
            total_samples += 1
            ious = torchvision.ops.box_iou(pred_box.unsqueeze(0), target_boxes)
            best_iou, best_target_idx = ious.max(dim=1)

            if best_iou >= iou_threshold and pred_label == target_labels[best_target_idx]:
                total_correct += 1

    accuracy = total_correct / total_samples if total_samples > 0 else 0
    return accuracy
