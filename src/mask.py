import rasterio
from rasterio.merge import merge
import numpy as np

def mask_clouds(stacked_img_path, qa_path, filename):
    with rasterio.open(stacked_img_path) as src:
        img = src.read().astype(np.float32)  # <-- Convert to float
        meta = src.meta
    
    with rasterio.open(qa_path) as qa_src:
        qa = qa_src.read(1)
    
    cloud_mask = (qa & (1 << 3)) != 0  # Cloud bit 3
    shadow_mask = (qa & (1 << 4)) != 0  # Shadow bit 4
    mask = cloud_mask | shadow_mask
    
    img_masked = np.where(mask, np.nan, img)  # Now works with float
    
    # Update metadata to float32
    meta.update(dtype=rasterio.float32)
    
    # Save masked image
    with rasterio.open(filename, 'w', **meta) as dst:
        dst.write(img_masked)
        
mask_clouds('../stacked_landsat/116047.tif', '../landsat/LC08_L2SP_116047_20201224_20210310_02_T1_QA_PIXEL.TIF', '../stacked_landsat/masked/116047.tif')
mask_clouds('../stacked_landsat/116048.tif', '../landsat/LC08_L2SP_116048_20201224_20210310_02_T1_QA_PIXEL.TIF', '../stacked_landsat/masked/116048.tif')
mask_clouds('../stacked_landsat/116049.tif', '../landsat/LC08_L2SP_116049_20201224_20210310_02_T1_QA_PIXEL.TIF', '../stacked_landsat/masked/116049.tif')
mask_clouds('../stacked_landsat/117048.tif', '../landsat/LC08_L2SP_117048_20201215_20210314_02_T1_QA_PIXEL.TIF', '../stacked_landsat/masked/117048.tif')
mask_clouds('../stacked_landsat/117049.tif', '../landsat/LC08_L2SP_117049_20201231_20210308_02_T1_QA_PIXEL.TIF', '../stacked_landsat/masked/117049.tif')