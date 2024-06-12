import torch
import numpy as np

def least_confidence(scores):
    """Calculates the least confidence score for each bounding box."""
    if scores.nelement() == 0:  # Check if tensor is empty
        return torch.tensor([0.0])  # Return a single 0 if empty (or another default value)
    else:
        return 1 - scores.max(dim=1)[0] # 1 - max confidence if not empty

def margin_confidence(scores):
    """Calculates the margin between the two highest confidence scores."""
    top_two = torch.topk(scores, 2, dim=1)[0]
    return top_two[:, 0] - top_two[:, 1]  # Difference between top two

def entropy(scores):
    """Calculates the entropy of the confidence scores."""
    normalized_scores = scores / scores.sum(dim=1, keepdim=True)
    return -torch.sum(normalized_scores * torch.log(normalized_scores), dim=1)

def uncertainty_sampling(model, image, device, uncertainty_method="least_confidence"):
    """Scores the unlabeled images based on the chosen uncertainty method."""
    model.eval()
    image_tensor = image.unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image_tensor)[0]  # Get predictions for the first (and only) image

    # Apply score thresholding (optional, but recommended)
    keep_idx = outputs['scores'] >= 0.5  # Adjust threshold as needed
    scores = outputs["scores"][keep_idx]
    boxes = outputs['boxes'][keep_idx]

    # Calculate uncertainty scores based on the chosen method
    if uncertainty_method == "least_confidence":
        uncertainty_scores = least_confidence(scores)
    elif uncertainty_method == "margin_confidence":
        uncertainty_scores = margin_confidence(scores)
    elif uncertainty_method == "entropy":
        uncertainty_scores = entropy(scores)
    else:
        raise ValueError(f"Unsupported uncertainty method: {uncertainty_method}")

    # You might want to aggregate the uncertainty scores across all boxes in an image
    # For example, take the mean or max uncertainty:
    image_score = uncertainty_scores.mean().item()  # Or .max()

    return image_score
