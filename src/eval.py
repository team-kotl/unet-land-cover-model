import rasterio
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from patchify import patchify
import segmentation_models_pytorch as smp
from sklearn.metrics import confusion_matrix, classification_report
import os

# Configuration (should match training)
PATCH_SIZE = 256
BATCH_SIZE = 8
IGNORE_INDEX = 255
NUM_CLASSES = 8
CLASS_NAMES = ['Water', 'Crops', 'Trees', 'Flooded Vegetation', 
              'Built Structures', 'Crops', 'Bare Ground', 'Rangeland']

# Reuse the same preprocessing functions from training
original_classes = [0, 1, 2, 5, 7, 8, 10, 11]
class_mapping = {old: new for new, old in enumerate(original_classes)}
max_class = max(original_classes)
lookup_table = np.full(max_class + 2, IGNORE_INDEX, dtype=np.int64)
for old, new in class_mapping.items():
    lookup_table[old] = new

def remap_classes(mask_array):
    clipped = np.clip(mask_array, 0, max_class + 1)
    return lookup_table[clipped]

def load_and_preprocess(image_path, mask_path):
    with rasterio.open(image_path) as src:
        image = src.read().transpose(1, 2, 0) / 10000.0
    with rasterio.open(mask_path) as src:
        mask = remap_classes(src.read(1))

    h, w = image.shape[:2]
    pad_h = (PATCH_SIZE - (h % PATCH_SIZE)) % PATCH_SIZE
    pad_w = (PATCH_SIZE - (w % PATCH_SIZE)) % PATCH_SIZE

    image_padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    mask_padded = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=IGNORE_INDEX)

    return image_padded, mask_padded

class LandCoverDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks
        self.valid_indices = [i for i, m in enumerate(masks) if not np.all(m == IGNORE_INDEX)]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        image = torch.tensor(self.images[actual_idx].transpose(2, 0, 1), dtype=torch.float32)
        mask = torch.tensor(self.masks[actual_idx], dtype=torch.long)
        return image, mask

def calculate_metrics(preds, labels, num_classes, ignore_index):
    # Filter out ignore index
    mask = labels != ignore_index
    preds = preds[mask]
    labels = labels[mask]

    # Calculate confusion matrix
    cm = confusion_matrix(labels.flatten().cpu().numpy(),
                         preds.flatten().cpu().numpy(),
                         labels=list(range(num_classes)))

    # Calculate metrics
    accuracy = np.diag(cm).sum() / cm.sum()
    iou = np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
    f1 = 2 * np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0))

    return {
        'confusion_matrix': cm,
        'overall_accuracy': accuracy,
        'iou': iou,
        'f1_score': f1
    }

def visualize_sample(image, true_mask, pred_mask, save_path):
    # Convert tensors to numpy arrays
    image = image.cpu().numpy().transpose(1, 2, 0)[:, :, [3,2,1]]  # RGB visualization
    true_mask = true_mask.cpu().numpy()
    pred_mask = pred_mask.cpu().numpy()

    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(np.clip(image * 3, 0, 1))  # Adjust visualization scaling
    plt.title('Input Image')
    
    plt.subplot(1, 3, 2)
    plt.imshow(true_mask, vmin=0, vmax=NUM_CLASSES-1)
    plt.title('Ground Truth')
    
    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask, vmin=0, vmax=NUM_CLASSES-1)
    plt.title('Prediction')
    
    plt.savefig(save_path)
    plt.close()

def evaluate_model(model_path, image_path, mask_path, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    image, mask = load_and_preprocess(image_path, mask_path)
    
    # Extract patches
    image_patches = patchify(image, (PATCH_SIZE, PATCH_SIZE, 4), step=PATCH_SIZE).reshape(-1, PATCH_SIZE, PATCH_SIZE, 4)
    mask_patches = patchify(mask, (PATCH_SIZE, PATCH_SIZE), step=PATCH_SIZE).reshape(-1, PATCH_SIZE, PATCH_SIZE)
    
    # Create dataset and loader
    dataset = LandCoverDataset(image_patches, mask_patches)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Load model
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=4,
        classes=NUM_CLASSES
    ).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Store predictions and labels
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(loader):
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            # Save visualizations for first batch
            if i == 0:
                for j in range(min(4, len(images))):  # Save 4 samples
                    visualize_sample(
                        images[j],
                        masks[j],
                        preds[j],
                        os.path.join(output_dir, f'sample_{j}.png')
                    )
            
            all_preds.append(preds)
            all_labels.append(masks.to(device))
    
    # Concatenate all results
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    # Calculate metrics
    metrics = calculate_metrics(all_preds, all_labels, NUM_CLASSES, IGNORE_INDEX)
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}\n\n")
        f.write("Class-wise Metrics:\n")
        f.write("{:<15} {:<10} {:<10} {:<10}\n".format("Class", "IoU", "F1", "Support"))
        for i in range(NUM_CLASSES):
            support = metrics['confusion_matrix'][i].sum()
            f.write("{:<15} {:<10.4f} {:<10.4f} {:<10}\n".format(
                CLASS_NAMES[i],
                metrics['iou'][i],
                metrics['f1_score'][i],
                support
            ))
    
    print(f"Evaluation complete. Results saved to {output_dir}")

if __name__ == "__main__":
    evaluate_model(
        model_path="../models/unet_landcover.pth",
        image_path="../stacked/merged.tif",
        mask_path="../truth/landcover.tif",
        output_dir="../evaluation_results"
    )