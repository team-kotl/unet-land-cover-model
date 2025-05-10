import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Path to the GeoTIFF file
file_path = '../data/Cordillera_Composite_2016_2018-0000000000-0000000000.tif'

def normalize(band):
    """Normalize band to 0-1 scale using percentile stretching"""
    band_min, band_max = np.nanpercentile(band, (2, 98))
    return np.clip((band - band_min) / (band_max - band_min), 0, 1)

with rasterio.open(file_path) as src:
    # Read RGB bands (B4, B3, B2 correspond to bands 3, 2, 1)
    red = src.read(3)
    green = src.read(2)
    blue = src.read(1)

    # Normalize each band
    red_n = normalize(red)
    green_n = normalize(green)
    blue_n = normalize(blue)

    # Create RGB composite
    rgb = np.dstack((red_n, green_n, blue_n))

    # Plot the image
    plt.figure(figsize=(15, 15))
    plt.imshow(rgb)
    plt.title('True Color Composite (B4/B3/B2)')
    plt.axis('off')
    plt.show()