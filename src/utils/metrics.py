import torch
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score, precision_score, recall_score, f1_score

# Intersection over Union (IoU)
def calculate_iou(box1, box2):
    """Calculates IoU between two bounding boxes (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area
    return iou

# Mean Average Precision (mAP)
def calculate_map(y_true, y_pred, num_classes):
    """Calculates mAP for multi-class object detection."""
    average_precisions = []
    for class_id in range(num_classes):
        ap = average_precision_score(y_true[:, class_id], y_pred[:, class_id])
        average_precisions.append(ap)
    mAP = np.mean(average_precisions)
    return mAP

# Accuracy, Precision, Recall, F1-score (for classification tasks)
def calculate_classification_metrics(y_true, y_pred):
    """Calculates classification metrics for a binary or multi-class task."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return accuracy, precision, recall, f1
