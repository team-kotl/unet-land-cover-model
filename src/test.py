import os
import time
import logging
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
    WebFeatureService
)
from config import RAW_IMG_DIR, CRS_EPSG, RESOLUTION, MAX_TILE_SIZE
from sentinelhub import WmsRequest

def get_sentinelhub_config():
    config = SHConfig()
    config.sh_client_id = "6e2b6927-5e07-447a-96d6-e3849cc5adf9"
    config.sh_client_secret = "GUdBkVM3cSn2Vk06Yzs63PZNLLJPjJz4"
    config.instance_id = "de08d5c9-9287-4d4e-8e1e-4e088ba2d50e"
    return config

def validate_coverage(bbox):
    wms_request = WmsRequest(
        data_collection=DataCollection.SENTINEL2_L2A,
        layer='TRUE-COLOR-S2-L1C',
        bbox=bbox,
        time=("2016-04-08", "2018-12-31"),
        width=100,
        height=100,
        config=get_sentinelhub_config()
    )
    return len(wms_request.get_dates())

init_bbox = BBox([120.464663, 16.179283, 121.659416, 18.537701], crs=CRS.WGS84)
web_mercator_bbox = init_bbox.transform(CRS(3857))

print(f"{validate_coverage(web_mercator_bbox)} images")