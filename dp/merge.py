import rasterio
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT
import glob
import os

def merge_tiles(tile_dir, output_path, target_crs=None):
    """Merge tiles with CRS handling"""
    tile_paths = glob.glob(os.path.join(tile_dir, "*.tif"))
    
    if not tile_paths:
        raise ValueError("No TIFF files found")

    # Open first tile to get default CRS if not specified
    with rasterio.open(tile_paths[0]) as src:
        target_crs = target_crs or src.crs

    # Open all tiles with CRS conversion
    src_files = []
    for fp in tile_paths:
        try:
            src = rasterio.open(fp)
            if src.crs != target_crs:
                print(f"Reprojecting {os.path.basename(fp)} from {src.crs} to {target_crs}")
                src = WarpedVRT(src, crs=target_crs)
            src_files.append(src)
        except Exception as e:
            print(f"Skipping {fp}: {str(e)}")
            continue

    # Merge with CRS consistency
    mosaic, transform = merge(
        src_files,
        method='max',
        nodata=0,
        res=(10, 10)  # Force 10m resolution
    )

    # Write output
    meta = src_files[0].meta.copy()
    meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": transform,
        "crs": target_crs,
        "driver": "GTiff"
    })

    with rasterio.open(output_path, "w", **meta) as dest:
        dest.write(mosaic)

    for src in src_files:
        src.close()

    print(f"Merged {len(src_files)} tiles to {output_path}")

# Usage - specify target CRS if needed
merge_tiles(
    tile_dir="../sentinel_downloads",
    output_path="../merged/merged_image.tif",
    target_crs="EPSG:32651"  # UTM zone for Philippines
)