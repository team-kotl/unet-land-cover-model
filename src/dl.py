from pathlib import Path
from sentinelhub import BBox, CRS
from data_processing import download_region

# Cordillera Administrative Region bounding box=
init_bbox = BBox([120.464663, 16.179283, 121.659416, 18.537701], crs=CRS.WGS84)
web_mercator_bbox = init_bbox.transform(CRS(3857))
time_interval = ("2016-04-08", "2018-12-31")

download_region(
    bbox=web_mercator_bbox,
    time_interval=time_interval
)
print("Download completed! Check data/raw/images directory")