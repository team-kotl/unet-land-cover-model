import rasterio
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from patchify import patchify
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp

# Configuration
PATCH_SIZE = 256
BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
IGNORE_INDEX = 255
NUM_CLASSES = 8

# Original class values and mapping
original_classes = [0, 1, 2, 5, 7, 8, 10, 11]
class_mapping = {old: new for new, old in enumerate(original_classes)}
max_class = max(original_classes)

# Create lookup table for remapping
lookup_table = np.full(max_class + 2, IGNORE_INDEX, dtype=np.int64)
for old, new in class_mapping.items():
    lookup_table[old] = new

def remap_classes(mask_array):
    """Remap classes using vectorized operations"""
    clipped = np.clip(mask_array, 0, max_class + 1)
    return lookup_table[clipped]

def load_and_preprocess(image_path, mask_path):
    """Load and preprocess Landsat imagery and mask"""
    with rasterio.open(image_path) as src:
        image = src.read().transpose(1, 2, 0).astype(np.float32)
        num_channels = image.shape[2]
        image = image * 0.0000275 - 0.2

    with rasterio.open(mask_path) as src:
        mask = remap_classes(src.read(1))

    # Get original dimensions
    h_img, w_img = image.shape[:2]
    h_msk, w_msk = mask.shape[:2]

    # Calculate required padding for both dimensions
    def calculate_padding(dim, patch_size):
        return (patch_size - (dim % patch_size)) % patch_size

    # Calculate combined required dimensions
    required_h = max(h_img, h_msk)
    required_w = max(w_img, w_msk)

    # Calculate padding to reach required dimensions and make divisible by PATCH_SIZE
    pad_h = (PATCH_SIZE - (required_h % PATCH_SIZE)) % PATCH_SIZE
    pad_w = (PATCH_SIZE - (required_w % PATCH_SIZE)) % PATCH_SIZE
    final_h = required_h + pad_h
    final_w = required_w + pad_w

    # Pad image and mask to final dimensions
    image_padded = np.pad(image, 
                         ((0, final_h - h_img), (0, final_w - w_img), (0, 0)), 
                         mode='reflect')
    mask_padded = np.pad(mask, 
                        ((0, final_h - h_msk), (0, final_w - w_msk)), 
                        mode='constant', constant_values=IGNORE_INDEX)

    return image_padded, mask_padded, num_channels

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

def calculate_iou(preds, labels, num_classes, ignore_index):
    mask = labels != ignore_index
    preds = preds[mask]
    labels = labels[mask]

    iou_per_class = []
    for class_id in range(num_classes):
        tp = ((preds == class_id) & (labels == class_id)).sum().item()
        fp = ((preds == class_id) & (labels != class_id)).sum().item()
        fn = ((preds != class_id) & (labels == class_id)).sum().item()
        
        union = tp + fp + fn
        iou = tp / union if union > 0 else 0.0
        iou_per_class.append(iou)
    
    return iou_per_class

def main():
    # Load data with Landsat preprocessing
    image, mask, num_channels = load_and_preprocess("../../stacked/merged-landsat.tif", "../../truth/landcover-30m.tif")
    
    # Validate class mappings
    unique = np.unique(mask)
    print("Unique values after remapping:", unique)
    assert all((0 <= c < NUM_CLASSES) or (c == IGNORE_INDEX) for c in unique), "Invalid class values"

    # Extract patches with dynamic channel handling
    image_patches = patchify(image, (PATCH_SIZE, PATCH_SIZE, num_channels), step=PATCH_SIZE).reshape(-1, PATCH_SIZE, PATCH_SIZE, num_channels)
    mask_patches = patchify(mask, (PATCH_SIZE, PATCH_SIZE), step=PATCH_SIZE).reshape(-1, PATCH_SIZE, PATCH_SIZE)

    # Split dataset
    X_train, X_temp, y_train, y_temp = train_test_split(image_patches, mask_patches, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Create datasets and dataloaders
    train_dataset = LandCoverDataset(X_train, y_train)
    val_dataset = LandCoverDataset(X_val, y_val)
    test_dataset = LandCoverDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, pin_memory=True)

    # Initialize model with dynamic in_channels
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=num_channels,
        classes=NUM_CLASSES
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, masks).item() * images.size(0)

        avg_train_loss = train_loss / len(train_dataset)
        avg_val_loss = val_loss / len(val_dataset)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "../../models/landsat_model.pth")

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}")

    # Final evaluation
    print("\nStarting final evaluation on test set...")
    model.load_state_dict(torch.load("../../models/landsat_model.pth"))
    model.eval()
    
    test_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            test_loss += loss.item() * images.size(0)
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(masks.cpu())

    # Calculate metrics
    test_loss = test_loss / len(test_dataset)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    mask = all_labels != IGNORE_INDEX
    accuracy = (all_preds[mask] == all_labels[mask]).float().mean()
    
    iou_per_class = calculate_iou(all_preds, all_labels, NUM_CLASSES, IGNORE_INDEX)

    print(f"\nTest Results:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print("IoU per class:")
    for class_id, iou in enumerate(iou_per_class):
        print(f"  Class {class_id}: {iou:.4f}")

if __name__ == "__main__":
    main()