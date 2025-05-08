import os
import rasterio
import numpy as np
from skimage.transform import resize

# Define paths
DATA_DIR = "data"
RAW_IMAGES_DIR = "path/to/sentinel2_images"  # Update with actual path
RAW_MASKS_DIR = "path/to/corine_raster"     # Update with actual path
PATCH_SIZE = (256, 256)
NUM_CLASSES = 10  # Adjust based on simplified classes

# Simplified class mapping (example: reduce 44 Corine classes to 10)
CLASS_MAPPING = {
    # Add your mapping, e.g., 1: 0 (urban), 2: 1 (forest), etc.
}

def load_and_align(image_path, mask_path):
    with rasterio.open(image_path) as src_img, rasterio.open(mask_path) as src_mask:
        image = src_img.read([1, 2, 3]).transpose((1, 2, 0))  # RGB bands
        mask = src_mask.read(1)
        # Ensure same spatial extent and resolution (resampling if needed)
        if image.shape[:2] != mask.shape:
            mask = resize(mask, image.shape[:2], preserve_range=True, order=0).astype(np.uint8)
        return image, mask

def tile_image(image, mask, patch_size):
    h, w = image.shape[:2]
    tiles = []
    for i in range(0, h, patch_size[0]):
        for j in range(0, w, patch_size[1]):
            if i + patch_size[0] <= h and j + patch_size[1] <= w:
                img_tile = image[i:i+patch_size[0], j:j+patch_size[1]]
                mask_tile = mask[i:i+patch_size[0], j:j+patch_size[1]]
                tiles.append((img_tile, mask_tile))
    return tiles

def save_tiles(tiles, split, idx_start):
    for idx, (img, mask) in enumerate(tiles, idx_start):
        split_dir = os.path.join(DATA_DIR, split)
        os.makedirs(os.path.join(split_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(split_dir, "masks"), exist_ok=True)
        np.save(os.path.join(split_dir, "images", f"img_{idx}.npy"), img)
        np.save(os.path.join(split_dir, "masks", f"mask_{idx}.npy"), mask)

# Process data
image_files = [os.path.join(RAW_IMAGES_DIR, f) for f in os.listdir(RAW_IMAGES_DIR) if f.endswith(".tif")]
mask_files = [os.path.join(RAW_MASKS_DIR, f) for f in os.listdir(RAW_MASKS_DIR) if f.endswith(".tif")]

all_tiles = []
for img_path, mask_path in zip(image_files, mask_files):
    image, mask = load_and_align(img_path, mask_path)
    tiles = tile_image(image, mask, PATCH_SIZE)
    all_tiles.extend(tiles)

# Split into train, val, test (e.g., 70-20-10)
np.random.shuffle(all_tiles)
train_split = int(0.7 * len(all_tiles))
val_split = int(0.9 * len(all_tiles))

save_tiles(all_tiles[:train_split], "train", 0)
save_tiles(all_tiles[train_split:val_split], "val", train_split)
save_tiles(all_tiles[val_split:], "test", val_split)

print("Data preprocessing completed.")