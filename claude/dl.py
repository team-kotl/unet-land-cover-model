import ee

# Authenticate and initialize Earth Engine (uncomment and run once if not already done)
ee.Authenticate()
ee.Initialize(project='helical-sanctum-451207-m5')

# Define the Area of Interest (AOI) using a bounding box in WGS84
aoi = ee.Geometry.Rectangle([120.464663, 16.179283, 121.659416, 18.537701])

# Filter Sentinel-2 L2A collection (surface reflectance)
collection = ee.ImageCollection('COPERNICUS/S2_SR') \
    .filterBounds(aoi) \
    .filterDate('2016-04-08', '2018-12-31') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))  # Filter for low cloud cover

# Function to mask clouds using the SCL band
def maskS2clouds(image):
    scl = image.select('SCL')
    # Mask clouds, cloud shadows, and snow (SCL values 3, 8, 9, 10, 11)
    cloud_mask = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10)).Or(scl.eq(11))
    # Select only the desired bands and apply the mask
    return image.select(['B2', 'B3', 'B4', 'B8']).updateMask(cloud_mask.Not())

# Apply the cloud mask and band selection to the collection
masked_collection = collection.map(maskS2clouds)

# Create a median composite
composite = masked_collection.median()

# Export the composite to Google Drive
task = ee.batch.Export.image.toDrive(
    image=composite,
    description='Cordillera_Sentinel2_Composite_2016_2018',
    folder='GEE_Exports',  # Folder in Google Drive
    fileNamePrefix='Cordillera_Composite_2016_2018',
    region=aoi,
    scale=10,  # 10m resolution
    crs='EPSG:32651',  # UTM Zone 51N
    maxPixels=1e10  # Increase if pixel limit errors occur
)

# Start the export
task.start()

print("Export task started. Check your Google Drive folder 'GEE_Exports' for the file.")