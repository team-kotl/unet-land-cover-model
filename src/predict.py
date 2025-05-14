import torch
import rasterio
import numpy as np
from skimage.util import view_as_blocks


def predict(model, image_path, patch_size=256):
    model.eval()

    # Load image
    with rasterio.open(image_path) as src:
        image = src.read()  # Shape: (C, H, W)
        meta = src.meta
        transform = src.transform

    # Pad image to make dimensions divisible by patch_size
    _, H, W = image.shape
    pad_H = (patch_size - H % patch_size) % patch_size
    pad_W = (patch_size - W % patch_size) % patch_size
    image_padded = np.pad(image, [(0, 0), (0, pad_H), (0, pad_W)], mode="constant")

    # Split into spatial patches (keep channels)
    # New shape: (num_patches_h, num_patches_w, C, patch_size, patch_size)
    patches = view_as_blocks(
        image_padded.transpose(1, 2, 0), (patch_size, patch_size, image.shape[0])
    )

    # Reshape to (num_patches, C, patch_size, patch_size)
    patches = patches.reshape(-1, image.shape[0], patch_size, patch_size)

    # Predict patches
    preds = []
    with torch.no_grad():
        for patch in patches:
            patch_tensor = (
                torch.tensor(patch, dtype=torch.float32).unsqueeze(0).to("cuda")
            )
            output = model(patch_tensor)  # Remove ["out"] for SMP models
            pred = output.argmax(dim=1).cpu().numpy()
            preds.append(pred)

    # Reconstruct full prediction
    preds = np.stack(preds)
    n_h = image_padded.shape[1] // patch_size
    n_w = image_padded.shape[2] // patch_size
    preds = preds.reshape(n_h, n_w, patch_size, patch_size)
    prediction = preds.transpose(0, 2, 1, 3).reshape(
        image_padded.shape[1], image_padded.shape[2]
    )

    # Remove padding and save
    prediction = prediction[:H, :W].astype(np.uint8)

    meta.update({"count": 1, "dtype": "uint8", "height": H, "width": W})

    with rasterio.open("../predict/prediction.tif", "w", **meta) as dst:
        dst.write(prediction, 1)

if __name__ == "__main__":
    # Load model (ensure architecture matches training)
    model = torch.load("../models/unet_landcover_final.pth", map_location="cuda", weights_only=False)
    model.eval()
    predict(model, "../merged/final.tif")
