import rasterio
import numpy as np
import matplotlib.pyplot as plt

# file_path = '../stacked_landsat/masked/116049.tif'
file_path = '../truth/landcover_labels.tif'

# Band indices (0-based for 6-band stack: B2, B3, B4, B5, B6, B7)
red_band = 3   # NIR (B5)
green_band = 2  # Red (B4)
blue_band = 1   # Green (B3)

percentile = 2
epsilon = 1e-6  # Small value to prevent division by zero

def normalize_band(band, percentile):
    """Handle NaN and uniform bands"""
    band_non_nan = band[~np.isnan(band)]
    
    if len(band_non_nan) == 0:
        return np.zeros_like(band)
    
    low, high = np.nanpercentile(band_non_nan, (percentile, 100 - percentile))
    
    if high - low < epsilon:
        return np.zeros_like(band)
    
    band = np.clip(band, low, high)
    return (band - low) / (high - low + epsilon)

with rasterio.open(file_path) as src:
    # Read and scale reflectance values (Landsat SR is scaled by 0.0001)
    red = src.read(red_band + 1).astype(float) * 0.0001
    green = src.read(green_band + 1).astype(float) * 0.0001
    blue = src.read(blue_band + 1).astype(float) * 0.0001

    # Normalize bands
    red_norm = normalize_band(red, percentile)
    green_norm = normalize_band(green, percentile)
    blue_norm = normalize_band(blue, percentile)

    # Replace NaN (if any remain)
    rgb = np.nan_to_num(np.dstack((red_norm, green_norm, blue_norm)))

    # Plot
    plt.figure(figsize=(12, 12))
    plt.imshow(rgb)
    plt.axis('off')
    plt.show()