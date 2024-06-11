import torch
import os
import sys
import argparse
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.faster_rcnn import fasterrcnn_resnet18
from src.utils.dataloader import create_dataloader
from src.utils.engine import train_one_epoch, evaluate
from src.utils.get_optimizer import get_optimizer
from src.utils.get_scheduler import get_scheduler
from src.utils.logger import TrainingLogger
from src.utils.general_utils import read_config
from src.utils.label_panel import visualize_predictions


def get_args():
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--config", type=str, default="config/faster_rcnn18_config.yaml", help="Path to the config file")
    return parser.parse_args()

def main(config):
    global transform
    
    device = torch.device('cuda' if torch.cuda.is_available() and config["device"] == 'gpu' else 'cpu')
    num_classes = config["num_classes"]
    unlabeled_data_path = os.path.join(config["dataroot"], config["unlabel_data"])
    active_learning_epochs = config["active_learning_epochs"] 
    batch_size = config["batch_size"]
    score_threshold = config["score_threshold"]
    
    # Transform
    transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((config["img_height"], config["img_width"])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    # 1. Initialization (train initial model on labeled data)
    train_loader = create_dataloader(config, is_train=True)
    val_loader = create_dataloader(config, is_train=False)
    model = fasterrcnn_resnet18(num_classes=num_classes, pretrained=True, coco_model=True).to(device)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    logger = TrainingLogger(config)
    
    print(f"Warning: Active learning loop is not implemented yet. Training initial model...")
    for epoch in range(config["num_epochs"]):
        train_one_epoch(model, optimizer, train_loader, device, epoch, logger)
        scheduler.step()
        if (epoch + 1) % config.get("val_interval", 1) == 0:
            evaluate(model, val_loader, device, logger)  # Pass logger for logging

    print("Initial model training completed.")
    print("Starting active learning loop...")
    # 2. Active Learning Loop
    for round in range(active_learning_epochs):
        # a. Score images (using model uncertainty - least confidence)
        unlabeled_scores = score_unlabeled_images(model, unlabeled_data_path, device, score_threshold) 

        # b. Select images for labeling
        images_to_label = select_images(unlabeled_scores, batch_size)
        print("Images to label:", images_to_label)
        # c. Oracle labeling (simulate by saving model predictions)
        for image_path in images_to_label:
            image_name = os.path.basename(image_path)
            image = Image.open(image_path).convert('RGB') # Load image from unlabeled_data_path
            image = to_tensor(image)  # Convert PIL Image to Tensor
            image = transform(image)
            with torch.no_grad():
                predictions = model([image.to(device)])[0]
            # TODO: Save predictions as labels
            visualize_predictions(image_path, predictions, "labels.txt")  # Save predictions as labels
            
            # Move the image from unlabel to train folder (update the data structure)
            os.replace(image_path, os.path.join(config["dataroot"], config["train_data"], "images", os.path.basename(image_path)))
            os.replace(image_path.replace(".jpg", ".txt"), os.path.join(config["dataroot"], config["train_data"], "labels", os.path.basename(image_path).replace(".jpg", ".txt")))

        # e. Model Update (retrain on updated dataset)
        train_loader = create_dataloader(config, is_train=True)  # Recreate dataloader
        print(f"Round {round + 1}/{active_learning_epochs}: Retraining model...")
        for epoch in range(config["num_epochs"]):
            train_one_epoch(model, optimizer, train_loader, device, epoch, logger)
            scheduler.step()
            if (epoch + 1) % config.get("val_interval", 1) == 0:
                evaluate(model, val_loader, device, logger)


def score_unlabeled_images(model, unlabeled_data_path, device, score_threshold):
    global transform
    model.eval()
    image_scores = {}
    image_names = os.listdir(unlabeled_data_path)

    for image_name in tqdm(image_names, desc="Scoring unlabeled images"):
        image_path = os.path.join(unlabeled_data_path, image_name)
        # TODO: Load image and preprocess
        image = Image.open(image_path).convert('RGB') # Load image from unlabeled_data_path 
        image = to_tensor(image)  # Convert PIL Image to Tensor
        image = transform(image)
        with torch.no_grad():
            outputs = model([image.to(device)])[0]
        
        # Calculate least confidence (1 - max probability)
        least_confidence = 1 - outputs['scores'].max().item()

        # Filter by score threshold and check if any predictions exist
        if (outputs['scores'] >= score_threshold).any():
            image_scores[image_path] = least_confidence  

    return image_scores


def select_images(image_scores, batch_size):
    """Selects images for labeling based on the lowest confidence scores."""
    sorted_images = sorted(image_scores, key=image_scores.get)
    return sorted_images[:batch_size]


if __name__ == "__main__":
    args = get_args()
    config = read_config(args.config)
    main(config)
