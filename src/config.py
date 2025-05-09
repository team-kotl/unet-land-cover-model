from pathlib import Path
import albumentations as A
import torch

# Paths
DATA_DIR = Path("../data")
RAW_IMG_DIR = DATA_DIR / "raw/images_3857"
RAW_MASK_DIR = DATA_DIR / "raw/masks_3857"
PROCESSED_DIR = DATA_DIR / "processed"

# Projection
CRS_EPSG = 3857  # Web Mercator

# Model
NUM_CLASSES = 12  # Updated for 12 land cover types
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Sentinel Hub
RESOLUTION = 10  # Meters
MAX_TILE_SIZE = 2500  # Pixels

# Classes
CLASS_NAMES = [
    "Closed Forest", "Open Forest", "Mangrove Forest",
    "Brush/Shrubs", "Grassland", "Perennial Crop",
    "Annual Crop", "Open/Barren", "Built-up",
    "Marshland/Swamp", "Fishpond", "Inland Water"
]

# Data augmentation
train_transform = A.Compose([
    A.RandomRotate90(),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.RandomBrightnessContrast(p=0.2),
])