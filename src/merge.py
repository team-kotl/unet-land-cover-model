import numpy as np
import rasterio

band_paths = ['../raw/B02.jp2', '../raw/B03.jp2', '../raw/B04.jp2', '../raw/B08.jp2']
with rasterio.open(band_paths[0]) as src:
    meta = src.meta
meta.update(count=len(band_paths))

with rasterio.open('../stacked.tif', 'w', **meta) as dst:
    for idx, path in enumerate(band_paths, start=1):
        with rasterio.open(path) as src:
            dst.write(src.read(1), idx)