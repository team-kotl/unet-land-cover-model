import rasterio
import numpy as np
import matplotlib.pyplot as plt

def show_rgb(image_path, lower_percent=2, upper_percent=98):
    """Display RGB composite with automatic contrast adjustment"""
    with rasterio.open(image_path) as src:
        # Read RGB bands (B04, B03, B02)
        rgb = src.read([3, 2, 1]).astype(np.float32)
        
        # Calculate percentiles for contrast stretching
        p_low, p_high = np.percentile(rgb, [lower_percent, upper_percent], axis=(1,2))
        
        # Normalize each band separately
        rgb_norm = np.zeros_like(rgb)
        for i in range(3):
            band = rgb[i]
            band = np.clip(band, p_low[i], p_high[i])
            rgb_norm[i] = (band - p_low[i]) / (p_high[i] - p_low[i]) * 255
            
        # Convert to uint8 and transpose to (H,W,C)
        rgb_uint8 = np.clip(rgb_norm, 0, 255).astype(np.uint8).transpose(1, 2, 0)
    
    # Display
    plt.figure(figsize=(12, 8))
    plt.imshow(rgb_uint8)
    plt.axis('off')
    plt.title(f"RGB Composite ({lower_percent}-{upper_percent}% contrast stretch)")
    plt.show()

# Usage with different stretch ranges
show_rgb("data/raw/images_3857/tile_13425000_1875000.tif")  # Default 2-98%
show_rgb("data/raw/images_3857/tile_13425000_1875000.tif", 5, 95)  # More conservative
show_rgb("data/raw/images_3857/tile_13425000_1875000.tif", 0, 100)  # Full dynamic range