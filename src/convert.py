import geopandas as gpd
import rasterio
import numpy as np
from rasterio.features import rasterize

# Open shapefile
gdf = gpd.read_file("../truth/land_cover_map_car_2020.shp")

# Get the shape of the Landsat raster (to match)
with rasterio.open("../merged/landsat.tif") as src:
    transform = src.transform
    out_shape = src.shape

# Rasterize
landcover_raster = rasterize(
    shapes=[(geom, class_id) for geom, class_id in zip(gdf.geometry, gdf['class_id'])],
    out_shape=out_shape,
    transform=transform,
    fill=0,  # Background value
    dtype=np.uint8 
)

# Save rasterized labels
with rasterio.open(
    "landcover_labels.tif",
    'w',
    driver='GTiff',
    height=out_shape[0],
    width=out_shape[1],
    count=1,
    dtype=np.uint8,
    crs=gdf.crs,
    transform=transform,
) as dst:
    dst.write(landcover_raster, 1)