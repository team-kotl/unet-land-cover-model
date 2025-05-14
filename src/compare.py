# compare.py
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

def compare(pred_path, gt_path, num_classes, nodata_value=None):
    # Load data
    with rasterio.open(pred_path) as src:
        pred = src.read(1)
    with rasterio.open(gt_path) as src:
        gt = src.read(1)
    
    # Handle nodata
    if nodata_value is not None:
        mask = (gt != nodata_value)
        gt = gt[mask]
        pred = pred[mask]
    
    # Metrics
    print(classification_report(gt, pred))
    cm = confusion_matrix(gt, pred)
    
    # Visualization
    plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.imshow(gt, cmap='tab20', vmax=num_classes)
    plt.title("Ground Truth")
    plt.subplot(132)
    plt.imshow(pred, cmap='tab20', vmax=num_classes)
    plt.title("Prediction")
    plt.subplot(133)
    plt.imshow(gt != pred, cmap='Reds')
    plt.title("Disagreement")
    plt.show()

if __name__ == "__main__":
    compare("../predict/prediction.tif", 
           "../merged/final.tif",
           num_classes=22,
           nodata_value=0)