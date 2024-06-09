import torch
from torchvision import transforms

class ImageTransformer:
    def __init__(self, config: dict):
        """Initializes the image transformer based on configuration parameters.

        Args:
            config: A dictionary containing configuration parameters.
        """
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config["img_height"], config["img_width"])),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config["img_height"], config["img_width"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, image: torch.Tensor, is_train: bool) -> torch.Tensor:
        """Applies the appropriate transformation to the image.

        Args:
            image: The image tensor to be transformed.
            is_train: A flag indicating whether the image is for training (True) or testing (False).

        Returns:
            The transformed image tensor.
        """
        if is_train:
            return self.train_transform(image)
        else:
            return self.test_transform(image)
