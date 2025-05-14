import rasterio
from rasterio.merge import merge
import numpy as np

# List of bands for one scene (adjust filenames)
band_paths = [
    '../landsat/LC08_L2SP_117049_20201231_20210308_02_T1_SR_B2.TIF', 
    '../landsat/LC08_L2SP_117049_20201231_20210308_02_T1_SR_B3.TIF', 
    '../landsat/LC08_L2SP_117049_20201231_20210308_02_T1_SR_B4.TIF', 
    '../landsat/LC08_L2SP_117049_20201231_20210308_02_T1_SR_B5.TIF', 
    '../landsat/LC08_L2SP_117049_20201231_20210308_02_T1_SR_B6.TIF', 
    '../landsat/LC08_L2SP_117049_20201231_20210308_02_T1_SR_B7.TIF'
]

# Read and stack bands
with rasterio.open(band_paths[0]) as src:
    meta = src.meta
meta.update(count=len(band_paths))  # Update metadata for multiband

stacked_output = "../stacked_landsat/117049.tif"
with rasterio.open(stacked_output, 'w', **meta) as dst:
    for idx, band in enumerate(band_paths, start=1):
        with rasterio.open(band) as src:
            dst.write(src.read(1), idx)