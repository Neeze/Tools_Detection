import torch
import sys
import os
import numpy as np
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import shutil
from src.models.faster_rcnn import fasterrcnn_resnet18
from src.utils.dataloader import create_dataloader
from src.utils.engine import train_one_epoch, evaluate, load_model
from src.utils.get_optimizer import get_optimizer
from src.utils.get_scheduler import get_scheduler
from src.utils.logger import TrainingLogger
from src.utils.general_utils import read_config, summary
from src.utils.label_panel import visualize_predictions
from src.utils.sampling import uncertainty_sampling
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor
from PIL import Image
from tqdm import tqdm
import random

def get_args():
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--config", type=str, default="config/faster_rcnn18_config.yaml", help="Path to the config file")
    return parser.parse_args()

def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() and config["device"] == 'gpu' else 'cpu')
    num_classes = config["num_classes"]
    active_learning_epochs = config["active_learning_epochs"]
    unlabeled_data_path = os.path.join(config["dataroot"], config["unlabel_data"])
    sampling_num = config["sampling_num"]
    score_threshold = config["score_threshold"]
    active_learning_strategy = config["active_learning_strategy"]

    # Create image transformer
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    # 1. Initialization (train initial model on labeled data)
    train_loader = create_dataloader(config, is_train=True)
    val_loader = create_dataloader(config, is_train=False)
    
    model = fasterrcnn_resnet18(num_classes=config["num_classes"], pretrained=True, coco_model=True).to(device)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    
    if "pretrained_model_path" in config.keys():
        print(f"Loading pretrained model from {config['pretrained_model_path']}")
        model, optimizer, scheduler = load_model(model, optimizer, scheduler, config["pretrained_model_path"])
        
    print("Model Summary:")
    summary(model)
    
    logger = TrainingLogger(config)

    for epoch in range(config["num_epochs"]):
        train_one_epoch(model, optimizer, train_loader, device, epoch, logger)
        scheduler.step()
        if (epoch + 1) % config.get("val_interval", 1) == 0:
            evaluate(model, val_loader, device, logger)  # Pass logger for logging
    print("Finished initial training")
    print("Starting Active Learning")
    # 2. Active Learning Loop
    for _ in range(active_learning_epochs):
        print(f"Active Learning Epoch {_ + 1}/{active_learning_epochs}")
        # a. Score images
        print("Scoring unlabeled images")
        unlabeled_scores = score_unlabeled_images(model, unlabeled_data_path, device, score_threshold, active_learning_strategy, image_transform)
        # unlabeled_scores['data/unlabel/20240610_124649.jpg'] = 0.9
        # b. Select images for labeling
        print("Selecting images for labeling")
        if len( unlabeled_scores.keys()) > 0:
            images_to_label = select_images(unlabeled_scores, sampling_num, active_learning_strategy)
            # c. Oracle labeling (simulate by saving model predictions)
            print("Labeling images")
            for image_path in images_to_label:                
                image = Image.open(image_path).convert('RGB') 
                image = image.resize((config["img_width"], config["img_height"]))
                copy_image = to_tensor(image)  # Convert PIL Image to Tensor
                # image = image_transform(image)    # Apply transformations
                copy_image = copy_image.to(device)

                with torch.no_grad():
                    predictions = model([copy_image])[0]
                
                predictions['boxes'] = predictions['boxes'][:10]
                
                new_image_name = os.path.basename(image_path)
                new_label_name = os.path.basename(image_path).replace(".jpg", ".txt")
                new_label_path = os.path.join(config["dataroot"], config["train_data"], "labels", new_label_name)

                copy_image = copy_image.detach().cpu().numpy().transpose(1, 2, 0)
                visualize_predictions(copy_image, predictions, new_label_path)
                # Move the image and its label from unlabel to train folder 
                image.save(os.path.join(config["dataroot"], config["train_data"], "images", new_image_name))
                os.remove(image_path)
                # shutil.move(image_path, os.path.join(config["dataroot"], config["train_data"], "images", new_image_name))
                # shutil.move(image_path.replace(".jpg", ".txt"), os.path.join(config["dataroot"], config["train_data"], "labels", new_label_name))
        
        # d. Model Update (retrain on updated dataset)
        print("Retraining model")
        train_loader = create_dataloader(config, is_train=True)
        for epoch in range(config["num_epochs"]):
            print(f"Epoch {epoch + 1}/{config['num_epochs']}")
            train_one_epoch(model, optimizer, train_loader, device, epoch, logger)
            scheduler.step()
            if (epoch + 1) % config.get("val_interval", 1) == 0:
                evaluate(model, val_loader, device, logger)

def score_unlabeled_images(model, unlabeled_data_path, device, score_threshold, active_learning_strategy, image_transform):
    """
    Scores unlabeled images using the specified strategy.

    Args:
        model: The object detection model.
        unlabeled_data_path: Path to the directory containing unlabeled images.
        device: The device (CPU or GPU) to use for inference.
        score_threshold: Threshold for filtering out low-confidence predictions.
        active_learning_strategy: The strategy to use for scoring images.
        image_transform: The image transformation to apply.

    Returns:
        A dictionary of image paths and their corresponding uncertainty scores.
    """
    model.eval()
    image_scores = {}
    image_names = os.listdir(unlabeled_data_path)
    for image_name in tqdm(image_names, desc="Scoring unlabeled images"):
        # Check for valid image files
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        image_path = os.path.join(unlabeled_data_path, image_name)
        image = Image.open(image_path).convert("RGB")
        image = image_transform(image).to(device)
        # image = image_transform(image)
        image = image.to(device)

        image_score = uncertainty_sampling(model, image, device, uncertainty_method='least_confidence')

        # Filter by score threshold
        if active_learning_strategy == "least_confidence" or active_learning_strategy == "margin_confidence" or active_learning_strategy == "entropy":
            if image_score >= score_threshold:
                image_scores[image_path] = image_score
        elif active_learning_strategy == "random":
            image_scores[image_path] = image_score  # For random, we don't use threshold

    return image_scores

def select_images(image_scores, batch_size, active_learning_strategy):
    """Selects images for labeling based on the chosen strategy."""
    if active_learning_strategy == "least_confidence" or active_learning_strategy == "margin_confidence" or active_learning_strategy == "entropy":
        sorted_images = sorted(image_scores, key=image_scores.get, reverse=True)  # Sort in descending order for uncertainty
        return sorted_images[:batch_size]
    elif active_learning_strategy == "random":
        image_paths = list(image_scores.keys())
        return random.sample(image_paths, batch_size)   


if __name__ == "__main__":
    args = get_args()
    config = read_config(args.config)
    main(config)
