import torch
import sys
import os
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from PIL import Image
from tqdm import tqdm
from src.models.faster_rcnn import fasterrcnn_resnet18
from src.utils.get_optimizer import get_optimizer
from src.utils.get_scheduler import get_scheduler
from src.utils.general_utils import read_config
from src.utils.label_panel import visualize_predictions
from src.utils.engine import load_model
from src.utils.transform import ImageTransformer
from torchvision.transforms.functional import to_tensor
from torchvision import transforms

def get_args():
    parser = argparse.ArgumentParser(description="Predict model")
    parser.add_argument("--config", type=str, default="config/faster_rcnn18_config.yaml", help="Path to the config file")
    parser.add_argument("--image", type=str, help="Path to the image file for prediction")
    parser.add_argument("--test_dir", type=str, help="Path to the directory containing images for prediction")
    parser.add_argument("--save", type=str, help="Path to the directory to save predictions and visualizations")
    return parser.parse_args()

def main(config: dict, image_path: str = None, test_dir: str = None, save_dir: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() and config["device"] == "gpu" else "cpu")
    print(f"Using device: {device}")

    # Create model
    model = fasterrcnn_resnet18(num_classes=config["num_classes"], pretrained=True, coco_model=True).to(device)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    if "pretrained_model_path" in config.keys():
        model, optimizer, scheduler = load_model(model, optimizer, scheduler, config["pretrained_model_path"])

    if "pretrained_model_path" not in config.keys():
        print(f"Pretrained model is not provided! Please check config file!")
    else:
        model, optimizer, scheduler = load_model(model, optimizer, scheduler, config["pretrained_model_path"])
    
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    os.makedirs(save_dir, exist_ok=True)  # Create the save directory if it doesn't exist

    # Prediction for a single image
    if image_path:
        predict_and_save(model, image_path, image_transform, device, save_dir)

    # Prediction for multiple images in a directory
    if test_dir:
        for filename in tqdm(os.listdir(test_dir)):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(test_dir, filename)
                predict_and_save(model, image_path, image_transform, device, save_dir)


def predict_and_save(model, image_path, transformer, device, save_dir):
    """Predicts and saves the visualization and labels for a single image."""
    model.eval()
    image = Image.open(image_path).convert('RGB') 
    image_tensor = transformer(image).to(device)
    copy_image = to_tensor(image)
    copy_image = copy_image.detach().cpu().numpy().transpose(1, 2, 0)
    
    with torch.no_grad():
        prediction = model([image_tensor])[0]

    # Visualize and save predictions
    visualize_predictions(copy_image, prediction, os.path.join(save_dir, os.path.basename(image_path).replace(".jpg", ".txt")))  


if __name__ == "__main__":
    args = get_args()
    config = read_config(args.config)
    main(config, args.image, args.test_dir, args.save)
