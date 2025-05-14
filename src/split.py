import geopandas as gpd
import rasterio
import numpy as np
import os
import shutil
from pathlib import Path
from skimage.util import view_as_blocks
from sklearn.model_selection import train_test_split


# --------------------------------
# 2. Split into Patches + Train/Val/Test Split
# --------------------------------
def split_into_patches(image_path, label_path, output_dir, patch_size=256, 
                      train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    """
    Split raster into patches and save into train/val/test folders.
    """
    # Create directories
    for split in ['train', 'val', 'test']:
        Path(f"{output_dir}/{split}/images").mkdir(parents=True, exist_ok=True)
        Path(f"{output_dir}/{split}/labels").mkdir(parents=True, exist_ok=True)

    # Load data
    with rasterio.open(image_path) as src:
        landsat = src.read()  # (bands, H, W)
    with rasterio.open(label_path) as src:
        labels = src.read(1)  # (H, W)

    # Pad to divisible by patch_size
    bands, H, W = landsat.shape
    pad_H = (patch_size - H % patch_size) % patch_size
    pad_W = (patch_size - W % patch_size) % patch_size
    landsat_padded = np.pad(landsat, ((0,0), (0,pad_H), (0,pad_W)), mode='constant')
    labels_padded = np.pad(labels, ((0,pad_H), (0,pad_W)), mode='constant')

    # Generate patches
    landsat_patches = view_as_blocks(landsat_padded, (bands, patch_size, patch_size))
    landsat_patches = landsat_patches.reshape(-1, bands, patch_size, patch_size)
    label_patches = view_as_blocks(labels_padded, (patch_size, patch_size))
    label_patches = label_patches.reshape(-1, patch_size, patch_size)

    # Shuffle and split indices
    indices = np.arange(len(landsat_patches))
    np.random.seed(seed)
    np.random.shuffle(indices)

    # Split into train/val/test
    train_idx, temp_idx = train_test_split(indices, test_size=(1 - train_ratio), random_state=seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=test_ratio/(val_ratio + test_ratio), random_state=seed)

    # Save patches
    splits = {
        'train': train_idx,
        'val': val_idx,
        'test': test_idx
    }

    for split_name, idx in splits.items():
        for i in idx:
            img_patch = landsat_patches[i]
            lbl_patch = label_patches[i]
            np.save(f"{output_dir}/{split_name}/images/patch_{i}.npy", img_patch)
            np.save(f"{output_dir}/{split_name}/labels/patch_{i}.npy", lbl_patch)

# --------------------------------
# 3. Run Full Pipeline
# --------------------------------
if __name__ == "__main__":
    # Split into train/val/test patches
    split_into_patches(
        image_path="../merged/final.tif",
        label_path="../truth/landcover_labels.tif",
        output_dir="data",
        patch_size=256,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1
    )