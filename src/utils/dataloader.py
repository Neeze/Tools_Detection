import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from .transform import ImageTransformer
from .general_utils import read_config
from .dataset import ImageDataset

def create_dataloader(config: dict, is_train: bool) -> DataLoader:
    """Creates a PyTorch DataLoader for object detection.

    Args:
        config: A dictionary containing configuration parameters.
        is_train: A boolean flag indicating whether to create the dataloader for training (True) or validation/testing (False).

    Returns:
        A PyTorch DataLoader object.
    """

    image_transformer = ImageTransformer(config)

    if is_train:
        data_path = os.path.join(config["dataroot"], config["train_data"])
        dataset = ImageDataset(image_dir=os.path.join(data_path, "images"),
                           label_dir=os.path.join(data_path, "labels"),
                           transform=image_transformer, is_train=True)
    else:
        data_path = os.path.join(config["dataroot"], config["val_data"])
        dataset = ImageDataset(image_dir=os.path.join(data_path, "images"),
                           label_dir=os.path.join(data_path, "labels"),
                           transform=image_transformer, is_train=False)
    
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=is_train,
        num_workers=4,  # Adjust based on your hardware
        collate_fn=lambda batch: tuple(zip(*batch))  # Custom collate function for object detection
    )

    return dataloader


if __name__ == "__main__":
    config_file = "config/faster_rcnn18_config.yaml"
    config = read_config(config_file)

    train_loader = create_dataloader(config, is_train=True)
    val_loader = create_dataloader(config, is_train=False)
