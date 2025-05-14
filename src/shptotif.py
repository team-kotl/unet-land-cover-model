import geopandas as gpd
import rasterio
import numpy as np
from rasterio.features import rasterize

# Load shapefile
gdf = gpd.read_file("../truth/land_cover_map_car_2020.shp")

# --------------------------------------------
# FIX: Set the original CRS if missing
# --------------------------------------------
if gdf.crs is None:
    # Replace "EPSG:4326" with your shapefile's actual CRS
    original_crs = "EPSG:3857"
    gdf = gdf.set_crs(original_crs, allow_override=True)

# Get Landsat raster's CRS and metadata
with rasterio.open("../merged/final.tif") as src:
    target_crs = src.crs
    transform = src.transform
    out_shape = src.shape

# Reproject shapefile to match Landsat CRS
gdf = gdf.to_crs(target_crs)

# Rasterize
landcover_raster = rasterize(
    shapes=[(geom, class_id) for geom, class_id in zip(gdf.geometry, gdf['class_id'])],
    out_shape=out_shape,
    transform=transform,
    fill=0,
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
    crs=target_crs,
    transform=transform,
) as dst:
    dst.write(landcover_raster, 1)