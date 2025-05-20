import rasterio

band_paths = [
    "../../raw/landsat/B2.TIF",
    "../../raw/landsat/B3.TIF",
    "../../raw/landsat/B4.TIF",
    "../../raw/landsat/B5.TIF",
    "../../raw/landsat/B6.TIF",
    "../../raw/landsat/B7.TIF",
]
with rasterio.open(band_paths[0]) as src:
    meta = src.meta
meta.update(count=len(band_paths))

with rasterio.open("../../stacked/merged-landsat.tif", "w", **meta) as dst:
    for idx, path in enumerate(band_paths, start=1):
        with rasterio.open(path) as src:
            dst.write(src.read(1), idx)
