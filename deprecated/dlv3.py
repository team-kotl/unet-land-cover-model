"""
Sentinel-2 L2A Export with Fixed Data Types
"""
import ee
import geemap
import math

# Initialize GEE
ee.Authenticate()
ee.Initialize(project='helical-sanctum-451207-m5') # PROJECT ID HERE

# Configuration
AOI = ee.Geometry.BBox(120.464663, 16.179283, 121.659416, 18.537701)
DATE_RANGE = ('2016-04-08', '2018-12-31')
MAX_CLOUD_COVER = 30
EXPORT_SCALE = 10
MAX_PIXELS = 1e10
ERROR_MARGIN = 1  # Meters for geometry operations

# ----------------------------
# 1. Image Processing with Type Conversion
# ----------------------------
def process_image(image):
    """Convert all bands to consistent UInt16 type"""
    # Select bands and convert to UInt16
    optical_bands = image.select(['B2', 'B3', 'B4', 'B8']).toUint16()
    scl_band = image.select('SCL').toUint16()
    
    # Cloud masking
    cloud_mask = scl_band.eq(3).Or(scl_band.eq(8)).Or(scl_band.eq(9)).Not()
    
    # Apply mask and return all bands with consistent type
    return optical_bands.updateMask(cloud_mask).addBands(scl_band)

# ----------------------------
# 2. Tiling System
# ----------------------------
def calculate_tiles(aoi, scale=10, max_pixels=1e8):
    """Calculate optimal tile grid"""
    area = aoi.area(maxError=ERROR_MARGIN).divide(1e6).getInfo()  # km²
    px_per_km = 1000 / scale
    max_km_per_tile = math.sqrt(max_pixels) / px_per_km
    
    num_tiles = max(1, math.ceil(area / (max_km_per_tile ** 2)))
    return int(math.sqrt(num_tiles)) + 1

def create_tile_grid(aoi, grid_size):
    """Create tile grid"""
    bounds = aoi.bounds(maxError=ERROR_MARGIN)
    coords = bounds.getInfo()['coordinates'][0]
    min_lon, max_lon = coords[0][0], coords[2][0]
    min_lat, max_lat = coords[0][1], coords[2][1]
    
    lon_steps = (max_lon - min_lon) / grid_size
    lat_steps = (max_lat - min_lat) / grid_size
    
    tiles = []
    for i in range(grid_size):
        for j in range(grid_size):
            west = min_lon + i * lon_steps
            east = west + lon_steps
            south = min_lat + j * lat_steps
            north = south + lat_steps
            tiles.append(ee.Geometry.BBox(west, south, east, north))
    
    return tiles

# ----------------------------
# 3. Export Function
# ----------------------------
def export_large_area(image, aoi, prefix, scale=10):
    """Export with consistent data types"""
    grid_size = calculate_tiles(aoi, scale, MAX_PIXELS)
    tiles = create_tile_grid(aoi, grid_size)
    
    for idx, tile in enumerate(tiles):
        clipped_image = image.clip(tile)
        
        task = ee.batch.Export.image.toDrive(
            image=clipped_image,
            description=f'{prefix}_tile_{idx}',
            folder='GEE_Tiled_Exports_2',
            fileNamePrefix=f'{prefix}_tile_{idx}',
            scale=scale,
            region=tile,
            maxPixels=MAX_PIXELS,
            fileFormat='GeoTIFF',
            formatOptions={'cloudOptimized': True}
        )
        task.start()
        print(f"Started tile {idx+1}/{len(tiles)}")

# ----------------------------
# 4. Main Workflow
# ----------------------------
def main():
    # Get and process collection
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(AOI)
                  .filterDate(*DATE_RANGE)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD_COVER))
                  .map(process_image))  # Apply our processing function

    # Process each image
    image_list = collection.toList(collection.size())
    count = collection.size().getInfo()

    print(f"Processing {count} images")
    for i in range(count):
        image = ee.Image(image_list.get(i))
        date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
        print(f"Exporting {date}")
        export_large_area(image, AOI, f'S2_{date}', EXPORT_SCALE)

if __name__ == "__main__":
    main()