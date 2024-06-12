import torch
from torchvision import transforms
from typing import Tuple, Optional, Dict


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

    def __call__(self, image: torch.Tensor, target: Optional[Dict[str, torch.Tensor]] = None, is_train: bool = False) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Transforms image and bounding boxes.

        Args:
            image: The image tensor to be transformed.
            target: An optional dictionary containing the target data (e.g., bounding boxes).
            is_train: A flag indicating whether the image is for training (True) or testing (False).

        Returns:
            The transformed image tensor and the transformed target dictionary.
        """

        _, height, width = image.shape  # Correcting the variable names

        # Apply common transformations (resize, to tensor, normalize)
        image = self.transform(image)

        # Apply training augmentations if in training mode
        if is_train:
            image = self.train_augmentation(image)

        # Rescale the coordinates of the bounding boxes if target is provided
        if target is not None:
            transformed_size = torch.tensor(image.shape[1:], dtype=torch.float32)  # [h, w]
            boxes = target["boxes"]
            boxes = boxes.float()  # Ensuring boxes are float for correct scaling
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * transformed_size[1] / width  # Rescale x coordinates
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * transformed_size[0] / height  # Rescale y coordinates

            # Update the target dictionary with the transformed boxes
            target["boxes"] = boxes

        return image, target
