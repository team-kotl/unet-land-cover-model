import rasterio
from config import RAW_IMG_DIR, RAW_MASK_DIR

def check_alignment():
    """Verify image-mask alignment"""
    img_path = next(RAW_IMG_DIR.glob("*.tif"))
    mask_path = next(RAW_MASK_DIR.glob("*.tif"))
    
    with rasterio.open(img_path) as img, rasterio.open(mask_path) as mask:
        print("Image CRS:", img.crs)
        print("Mask CRS:", mask.crs)
        print("Image transform:", img.transform)
        print("Mask transform:", mask.transform)
        print("Image shape:", img.shape)
        print("Mask shape:", mask.shape)
        
        assert img.crs == mask.crs, "CRS mismatch"
        assert img.transform == mask.transform, "Transform mismatch"
        assert img.shape == mask.shape, "Size mismatch"

if __name__ == "__main__":
    check_alignment()