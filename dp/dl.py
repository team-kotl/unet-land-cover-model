import os
import datetime
import numpy as np
from sentinelhub import (
    SentinelHubRequest,
    DataCollection,
    MimeType,
    BBox,
    CRS,
    SHConfig,
    geometry,
    bbox_to_dimensions
)

# Configuration
CLIENT_ID = "6e2b6927-5e07-447a-96d6-e3849cc5adf9"
CLIENT_SECRET = "GUdBkVM3cSn2Vk06Yzs63PZNLLJPjJz4"
INSTANCE_ID = "de08d5c9-9287-4d4e-8e1e-4e088ba2d50e"  # From Sentinel Hub Dashboard
OUTPUT_DIR = "../sentinel_downloads"

RESOLUTION = 10  # meters per pixel
MAX_PIXELS = 2500  # Sentinel Hub limit

# Area of Interest (Cordillera Administrative Region)
AOI_BBOX = [120.464663, 16.179283, 121.659416, 18.537701]

def split_bbox(bbox, resolution=10):
    """Split large BBox into smaller tiles that comply with 2500x2500 limit"""
    original_bbox = BBox(bbox=bbox, crs=CRS.WGS84)
    original_size = bbox_to_dimensions(original_bbox, resolution=resolution)
    
    num_splits_x = np.ceil(original_size[0] / MAX_PIXELS).astype(int)
    num_splits_y = np.ceil(original_size[1] / MAX_PIXELS).astype(int)
    
    min_x, min_y, max_x, max_y = bbox
    tile_width = (max_x - min_x) / num_splits_x
    tile_height = (max_y - min_y) / num_splits_y
    
    tiles = []
    for i in range(num_splits_x):
        for j in range(num_splits_y):
            tile_min_x = min_x + i * tile_width
            tile_min_y = min_y + j * tile_height
            tile_max_x = min_x + (i + 1) * tile_width
            tile_max_y = min_y + (j + 1) * tile_height
            
            tile_bbox = BBox(
                bbox=[tile_min_x, tile_min_y, tile_max_x, tile_max_y],
                crs=CRS.WGS84
            )
            tiles.append(tile_bbox)
    
    print(f"Split AOI into {len(tiles)} tiles")
    return tiles

def download_tile(tile_bbox, time_interval, config):
    """Download data for a single tile"""
    evalscript = """
    //VERSION=3
    function setup() {
        return {
            input: ["B02", "B03", "B04", "B08", "SCL", "dataMask"],
            output: { bands: 6 }
        };
    }
    
    function evaluatePixel(sample) {
        return [
            sample.B02 * 2.5,
            sample.B03 * 2.5,
            sample.B04 * 2.5,
            sample.B08 * 2.5,
            sample.SCL,
            sample.dataMask
        ];
    }
    """
    
    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=time_interval
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=tile_bbox,
        size=bbox_to_dimensions(tile_bbox, resolution=RESOLUTION),
        config=config
    )
    
    return request.get_data()

def download_sentinel_data():
    config = SHConfig()
    config.sh_client_id = CLIENT_ID
    config.sh_client_secret = CLIENT_SECRET
    config.instance_id = INSTANCE_ID
    
    tiles = split_bbox(AOI_BBOX, RESOLUTION)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    current_date = datetime.date(2015, 12, 31)
    end_date = datetime.date(2015, 12, 31)
    
    while current_date <= end_date:
        month_start = current_date
        month_end = min(end_date, current_date + datetime.timedelta(days=31))
        
        try:
            all_data = []
            for idx, tile in enumerate(tiles):
                print(f"Processing tile {idx+1}/{len(tiles)} for {month_start.strftime('%Y-%m')}")
                tile_data = download_tile(tile, (month_start, month_end), config)
                if tile_data:
                    all_data.extend(tile_data)
            
            if all_data:
                filename = f"sentinel_{month_start.strftime('%Y%m')}.tiff"
                output_path = os.path.join(OUTPUT_DIR, filename)
                with open(output_path, "wb") as f:
                    f.write(all_data[0])  # Simplified example - adjust for multiple scenes
                print(f"Saved {filename}")
                
        except Exception as e:
            print(f"Error processing {month_start.strftime('%Y-%m')}: {str(e)}")
        
        current_date = month_end + datetime.timedelta(days=1)

if __name__ == "__main__":
    download_sentinel_data()
    print("Download completed. Check", OUTPUT_DIR)