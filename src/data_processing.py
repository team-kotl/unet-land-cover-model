import numpy as np
import rasterio
from rasterio.transform import from_origin
from sentinelhub import (
    SHConfig,
    SentinelHubRequest,
    bbox_to_dimensions,
    BBox,
    CRS,
    DataCollection,
    MimeType,
    UtmZoneSplitter,
    WebFeatureService,
    SentinelHubCatalog
)
from config import RAW_IMG_DIR, CRS_EPSG, RESOLUTION, MAX_TILE_SIZE

def get_sentinelhub_config():
    config = SHConfig()
    config.sh_client_id = "6e2b6927-5e07-447a-96d6-e3849cc5adf9"
    config.sh_client_secret = "GUdBkVM3cSn2Vk06Yzs63PZNLLJPjJz4"
    config.instance_id = "de08d5c9-9287-4d4e-8e1e-4e088ba2d50e"
    return config

def download_tile(tile_bbox, time_interval):
    """Download tile with cloud mask"""
    evalscript = """
    //VERSION=3
    function setup() {
    return {
        input: [{
        bands: ["B02", "B03", "B04", "B08"],
        units: "REFLECTANCE"
        }, {
        bands: ["SCL"],
        units: "DN"
        }],
        output: [
        { id: "default", bands: 4, sampleType: "UINT16" },
        { id: "cloud_mask", bands: 1, sampleType: "UINT8" }
        ]
    };
    }

    function evaluatePixel(samples) {
    return {
        default: [
        samples[0].B02 * 10000,
        samples[0].B03 * 10000,
        samples[0].B04 * 10000,
        samples[0].B08 * 10000
        ],
        cloud_mask: [isCloud(samples[1].SCL)]
    };
    }

    function isCloud(scl) {
    // 0: NODATA, 1: SATURATED, 2: DARK_FEATURE, 3: CLOUD_SHADOW, 
    // 8: CLOUD_MEDIUM_PROBA, 9: CLOUD_HIGH_PROBA, 10: THIN_CIRRUS
    return [3, 8, 9, 10].includes(scl) ? 1 : 0;
    }
    """

    try:
        request = SentinelHubRequest(
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=time_interval,
                    maxcc=0.3,  # More strict cloud filter
                )
            ],
            responses=[
                SentinelHubRequest.output_response("default", MimeType.TIFF),
                SentinelHubRequest.output_response("cloud_mask", MimeType.TIFF)
            ],
            bbox=tile_bbox,
            size=bbox_to_dimensions(tile_bbox, resolution=RESOLUTION),
            config=get_sentinelhub_config(),
            evalscript=evalscript
        )

        # Get and validate response
        response = request.get_data(show_progress=True)
        
        # Validate response structure
        if not response or len(response) < 2:
            raise ValueError(f"Empty response for stuff at {time_interval}")
            
        data_list, mask_list = request.get_data(show_progress=True)
        image_data = data_list[0]      # shape (H, W, 4)
        cloud_mask = mask_list[0].squeeze()  # shape (H, W) or (1, H, W)
        
        # Validate data content
        if np.all(image_data == 0):
            raise ValueError("Received all-zero image data")
            
        if image_data.shape[-1] != 4:  # Last dimension for HWC format
            raise ValueError(f"Unexpected image shape {image_data.shape}")
            
        return image_data, cloud_mask

    except Exception as e:
        print(f"Download failed for stuff: {str(e)}")
        raise

def save_geotiff(data, mask, bbox, image_path="../data/raw/images_3857", mask_path="../data/raw/masks_3857"):
    """Save image and cloud mask with proper metadata"""
    # Process image data
    if data.ndim == 3 and data.shape[2] == 4:
        data = np.transpose(data, (2, 0, 1))
    
    if data.shape[0] != 4:
        raise ValueError(f"Invalid image shape {data.shape}")
    
    # Process mask data
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    
    # Save image
    with rasterio.open(
        image_path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=4,
        dtype=data.dtype,
        crs=rasterio.crs.CRS.from_epsg(3857),
        transform=from_origin(bbox.min_x, bbox.max_y, RESOLUTION, -RESOLUTION),
    ) as dst:
        dst.write(data)
        dst.colorinterp = [
            rasterio.enums.ColorInterp.blue,
            rasterio.enums.ColorInterp.green,
            rasterio.enums.ColorInterp.red,
            rasterio.enums.ColorInterp.gray
        ]

    # Save mask
    with rasterio.open(
        mask_path,
        "w",
        driver="GTiff",
        height=mask.shape[0],
        width=mask.shape[1],
        count=1,
        dtype=mask.dtype,
        crs=rasterio.crs.CRS.from_epsg(3857),
        transform=from_origin(bbox.min_x, bbox.max_y, RESOLUTION, -RESOLUTION),
    ) as dst:
        dst.write(mask, 1)

def calculate_cloud_percentage(mask):
    """Calculate percentage of cloudy pixels"""
    cloud_pixels = np.count_nonzero(mask)
    total_pixels = mask.size
    return (cloud_pixels / total_pixels) * 100

def download_region(bbox, time_interval, max_cloud_percent=20):
    """Improved download function with tile validation"""
    RAW_IMG_DIR.mkdir(parents=True, exist_ok=True)

    # Configure WFS request with proper date field
    wfs = WebFeatureService(
        bbox,
        time_interval,
        data_collection=DataCollection.SENTINEL2_L2A,
        maxcc=0.5,
        config=get_sentinelhub_config()
    )

    # Find least cloudy image with correct date field
    best_tile = None
    min_clouds = 100
    
    for tile in wfs:
        properties = tile['properties']
        date_str = properties.get('date')  # Correct field name
            
        acquisition_date = date_str
        cloud_cover = properties.get('cloudCoverPercentage', 100)
        
        if cloud_cover < min_clouds:
            best_tile = {
                'date': acquisition_date,
                'cloud_cover': cloud_cover,
                'tile_id': properties.get('id', 'unknown')
            }
            min_clouds = cloud_cover

    if not best_tile:
        print("No suitable images found")
        return

    print(f"Best tile: {best_tile['date']} with {best_tile['cloud_cover']}% clouds")

    # Generate validated tile grid
    tile_size = MAX_TILE_SIZE * RESOLUTION
    min_x = np.floor(bbox.min_x / tile_size) * tile_size
    min_y = np.floor(bbox.min_y / tile_size) * tile_size
    max_x = np.ceil(bbox.max_x / tile_size) * tile_size
    max_y = np.ceil(bbox.max_y / tile_size) * tile_size

    # Create inclusive coordinate ranges
    x_coords = np.arange(min_x, max_x + tile_size, tile_size)
    y_coords = np.arange(min_y, max_y + tile_size, tile_size)

    for x in x_coords:
        for y in y_coords:
            tile_bbox = BBox([x, y, x + tile_size, y + tile_size], crs=CRS(3857))
            
            try:
                # Skip invalid coordinates
                if not is_valid_web_mercator(tile_bbox):
                    print(f"Skipping invalid tile {x}_{y}")
                    continue

                # Check actual data existence
                if not check_tile_data_exists(tile_bbox, best_tile['date']):
                    print(f"No data for tile {x}_{y} on {best_tile['date']}")
                    continue

                # Proceed with download
                width, height = bbox_to_dimensions(tile_bbox, RESOLUTION)
                if width < 10 or height < 10:
                    continue

                print(f"Processing tile {x}_{y}")
                image_data, cloud_mask = download_tile(tile_bbox, best_tile['date'])
                
                if image_data.size == 0:
                    raise ValueError("Received empty image data")

                cloud_percent = calculate_cloud_percentage(cloud_mask)
                if cloud_percent > max_cloud_percent:
                    print(f"Skipping tile {x}_{y} with {cloud_percent:.1f}% clouds")
                    continue

                save_geotiff(image_data, cloud_mask, tile_bbox)
                print(f"Saved tile {x}_{y}")

            except Exception as e:
                print(f"Failed tile {x}_{y}: {str(e)}")
                continue

def is_valid_web_mercator(bbox):
    """Validate Web Mercator coordinates"""
    max_coord = 20026376.39  # ~20037 km
    return all([
        abs(bbox.min_x) <= max_coord,
        abs(bbox.min_y) <= max_coord,
        abs(bbox.max_x) <= max_coord,
        abs(bbox.max_y) <= max_coord
    ])

def check_tile_data_exists(tile_bbox, date):
    """Verify data exists before download"""
    catalog = SentinelHubCatalog(config=get_sentinelhub_config())
    search_iterator = catalog.search(
        DataCollection.SENTINEL2_L2A,
        bbox=tile_bbox,
        time=(f"{date}T00:00:00Z", f"{date}T23:59:59Z"),
        limit=1
    )
    return any(search_iterator)