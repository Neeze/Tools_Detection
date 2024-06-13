import torch
import torchvision

def calculate_iou(box1, box2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    inter_xmin = torch.max(box1[0], box2[0])
    inter_ymin = torch.max(box1[1], box2[1])
    inter_xmax = torch.min(box1[2], box2[2])
    inter_ymax = torch.min(box1[3], box2[3])
    
    inter_area = torch.max(inter_xmax - inter_xmin, torch.tensor(0.0)) * torch.max(inter_ymax - inter_ymin, torch.tensor(0.0))
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area
    return iou

def calculate_precision_recall(targets, predictions, iou_threshold=0.5):
    """Calculates precision and recall for object detection."""
    tp, fp, fn = 0, 0, 0

    for target, prediction in zip(targets, predictions):
        target_boxes = target["boxes"]
        target_labels = target["labels"]
        pred_boxes = prediction["boxes"]
        pred_labels = prediction["labels"]
        pred_scores = prediction["scores"]

        matched = torch.zeros(len(target_boxes), dtype=torch.bool)

        for pred_box, pred_label, pred_score in zip(pred_boxes, pred_labels, pred_scores):
            ious = torch.tensor([calculate_iou(pred_box, target_box) for target_box in target_boxes])
            best_iou, best_idx = ious.max(dim=0)
            
            if best_iou >= iou_threshold and pred_label == target_labels[best_idx]:
                if not matched[best_idx]:
                    tp += 1
                    matched[best_idx] = True
                else:
                    fp += 1
            else:
                fp += 1

        fn += (~matched).sum().item()

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    return precision, recall

def calculate_map(targets, predictions, iou_threshold=0.5):
    """Calculates mean Average Precision (mAP) for object detection."""
    precisions, recalls = [], []

    for threshold in torch.arange(0.5, 1.0, 0.05):
        precision, recall = calculate_precision_recall(targets, predictions, iou_threshold=threshold)
        precisions.append(precision)
        recalls.append(recall)
    
    # Interpolate the precision-recall curve
    precisions = torch.tensor(precisions)
    recalls = torch.tensor(recalls)
    
    # Compute the Average Precision (AP) as the area under the precision-recall curve
    ap = torch.trapz(precisions, recalls)
    return ap.item()

def calculate_accuracy(targets, predictions, iou_threshold=0.5):
    """Calculates accuracy for object detection based on IoU threshold."""
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

if __name__ == "__main__":
    targets = [
        {"boxes": torch.tensor([[50, 50, 150, 150]], dtype=torch.float32), "labels": torch.tensor([1])}
    ]
    predictions = [
        {"boxes": torch.tensor([[45, 45, 155, 155]], dtype=torch.float32), "labels": torch.tensor([1]), "scores": torch.tensor([0.9])}
    ]
    num_classes = 2  # Assuming one class plus background

    # Calculate metrics
    map_value = calculate_map(targets, predictions, num_classes)
    precision, recall = calculate_precision_recall(targets, predictions)
    accuracy = calculate_accuracy(targets, predictions)

    print(f"mAP: {map_value:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
