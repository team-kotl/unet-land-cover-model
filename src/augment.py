from pathlib import Path
import torch
from torch.utils.data import Dataset
import albumentations as A
import numpy as np


class LandCoverDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.image_paths = sorted(self.image_dir.glob("*.npy"))
        self.label_paths = sorted(self.label_dir.glob("*.npy"))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image (H, W, C) and label (H, W)
        image = np.load(self.image_paths[idx]).astype(
            np.float32
        )  # Shape: (256, 256, 6)
        label = np.load(self.label_paths[idx]).astype(np.int64)  # Shape: (256, 256)

        # Apply augmentations to HWC format
        if self.transform:
            transformed = self.transform(image=image, mask=label)
            image = transformed["image"]  # Shape: (256, 256, 6)
            label = transformed["mask"]  # Shape: (256, 256)

        # Convert to channels-first (C, H, W) AFTER augmentation
        image = image.transpose(0, 1, 2)  # New shape: (6, 256, 256)
        
        image = np.nan_to_num(image, nan=0.0)  # Replace NaNs with 0
        image = np.clip(image, -1e3, 1e3)  # Clip extreme values

        # Ensure no extra dimensions (no squeeze needed)
        return (
            torch.tensor(image, dtype=torch.float32),  # Shape: (6, 256, 256)
            torch.tensor(label, dtype=torch.long),  # Shape: (256, 256)
        )

# Define augmentations
train_transform = A.Compose(
    [
        A.Rotate(limit=45, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ],
    additional_targets={"image": "image", "mask": "mask"},
)

val_transform = A.Compose([])  # No augmentation for validation
