import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from PIL import Image
from .transform import ImageTransformer  # Assuming the transform module is in the same directory


class ImageDataset(Dataset):
    def __init__(self, image_dir: str, label_dir: str, transform: ImageTransformer = None, is_train: bool = True):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.is_train = is_train
        self.image_filenames = os.listdir(image_dir)  # Get all image filenames

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_name = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_name)
        label_path = os.path.join(self.label_dir, image_name.replace('.jpg', '.txt'))  # Adjusting label filename
        
        # Read image
        image = Image.open(image_path).convert("RGB")  
        image = to_tensor(image)  # Convert PIL Image to Tensor

        boxes = []
        labels = []
        with open(label_path, "r") as f:
            for line in f:
                try:
                    class_label, *coords = line.strip().split()
                    xmin, ymin, xmax, ymax = map(float, coords)
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(int(class_label))
                except Exception as e:
                    print(f"Error processing line: {line}, Error: {e}")
        
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),  # Adjusted dtype
            "labels": torch.tensor(labels, dtype=torch.int64)  # Adjusted dtype
        }
        
        # Transform Image and Boxes
        if self.transform:
            image, target = self.transform(image, target, is_train=self.is_train)
        
        return image, target
