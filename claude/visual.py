import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings

# Configuration
# TILE_DIR = '../sentinel_downloads'
TILE_DIR = '../merged'
OUTPUT_DIR = '../visualizations'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# SCL Color Map
scl_colors = [
    [0, 0, 0],        # 0: NO_DATA
    [255, 0, 0],      # 1: SATURATED_DEFECTIVE
    [47, 79, 79],     # 2: DARK_FEATURE_SHADOW
    [220, 220, 220],  # 3: CLOUD_SHADOW
    [255, 165, 0],    # 4: VEGETATION
    [255, 255, 0],    # 5: BARE_SOIL
    [0, 100, 0],      # 6: WATER
    [148, 0, 211],    # 7: CLOUD_LOW_PROBA
    [255, 0, 255],    # 8: CLOUD_MEDIUM_PROBA
    [139, 37, 0],     # 9: CLOUD_HIGH_PROBA
    [100, 149, 237],  # 10: THIN_CIRRUS
    [152, 251, 152]   # 11: SNOW_ICE
]
scl_cmap = ListedColormap(np.array(scl_colors)/255)

def safe_normalize(band):
    """Normalization with NaN handling and type safety"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        band = band.astype(float)
        valid_pixels = band[~np.isnan(band)]
        
        if len(valid_pixels) == 0:
            return np.zeros_like(band)
            
        p2, p98 = np.nanpercentile(valid_pixels, [2, 98])
        if p98 <= p2:
            return np.zeros_like(band)
            
        normalized = (band - p2) / (p98 - p2)
        return np.clip(normalized, 0, 1)

def visualize_tile(tif_path):
    """Visualization with integer/float separation"""
    try:
        with rasterio.open(tif_path) as src:
            # Read bands with proper typing
            bands = src.read()
            
            # Separate optical bands (float) and SCL (integer)
            optical = bands[:4].astype(float)  # B2, B3, B4, B8
            scl = bands[4].astype(np.uint16)    # SCL band
            
            # Create masks
            valid_mask = scl != 0
            cloud_mask = np.isin(scl, [3, 8, 9, 10])
            
            # Apply valid mask to optical bands
            for i in range(optical.shape[0]):
                optical[i][~valid_mask] = np.nan
            
            # Create composites
            rgb = np.dstack([
                safe_normalize(optical[2]),  # Red (B4)
                safe_normalize(optical[1]),  # Green (B3)
                safe_normalize(optical[0])   # Blue (B2)
            ])
            
            fci = np.dstack([
                safe_normalize(optical[3]),  # NIR (B8)
                safe_normalize(optical[2]),  # Red (B4)
                safe_normalize(optical[1])   # Green (B3)
            ])
            
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(20, 20))
            
            # Plot RGB
            axes[0,0].imshow(rgb)
            axes[0,0].set_title('True Color (RGB)')
            axes[0,0].axis('off')
            
            # Plot False Color
            axes[0,1].imshow(fci)
            axes[0,1].set_title('False Color Infrared')
            axes[0,1].axis('off')
            
            # Plot SCL with valid mask
            scl_display = np.ma.masked_where(~valid_mask, scl)
            axes[1,0].imshow(scl_display, cmap=scl_cmap, vmin=0, vmax=11)
            axes[1,0].set_title('Scene Classification')
            axes[1,0].axis('off')
            
            # Plot Cloud Mask
            axes[1,1].imshow(cloud_mask, cmap='gray')
            axes[1,1].set_title('Cloud Mask (White=Clouds)')
            axes[1,1].axis('off')
            
            # Save output
            output_path = os.path.join(OUTPUT_DIR, 
                                     f"{os.path.splitext(os.path.basename(tif_path))[0]}_vis.png")
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
            print(f"Saved: {output_path}")
            
    except Exception as e:
        print(f"Error processing {tif_path}: {str(e)}")

def batch_visualize():
    """Process all tiles with error logging"""
    from tqdm import tqdm
    
    tif_files = []
    for root, _, files in os.walk(TILE_DIR):
        for file in files:
            if file.lower().endswith(('.tif', '.tiff')):
                tif_files.append(os.path.join(root, file))
    
    print(f"Found {len(tif_files)} tiles to process")
    for path in tqdm(tif_files, desc="Processing tiles"):
        visualize_tile(path)

if __name__ == "__main__":
    batch_visualize()