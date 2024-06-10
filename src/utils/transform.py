import torch
from torchvision import transforms
from typing import Tuple


class ImageTransformer:
    def __init__(self, config: dict):
        """Initializes the image transformer based on configuration parameters.

        Args:
            config: A dictionary containing configuration parameters.
        """
        self.config = config

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config["img_height"], config["img_width"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Augmentation specifically for training
        self.train_augmentation = transforms.RandomHorizontalFlip()

    def __call__(self, image: torch.Tensor, target: dict, is_train: bool) -> Tuple[torch.Tensor, dict]:
        """Transforms image and bounding boxes.

        Args:
            image: The image tensor to be transformed.
            target: A dictionary containing the target data (e.g., bounding boxes).
            is_train: A flag indicating whether the image is for training (True) or testing (False).

        Returns:
            The transformed image tensor and the transformed target dictionary.
        """

        _, witdh, height = image.shape

        # Apply common transformations (resize, to tensor, normalize)
        image = self.transform(image)
        boxes = target["boxes"]

            
        # Rescale the coordinates of the bounding boxes
        transformed_size = torch.tensor(image.shape[1::], dtype=torch.float32)  # [h, w]
        # boxes = torch.tensor(target["boxes"], dtype=torch.float32)  # Convert directly from target dict
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * transformed_size[1] / witdh # Normalize x coordinates
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * transformed_size[0] / height # Normalize y coordinates
    
        return image, target
