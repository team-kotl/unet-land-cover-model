import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from torch.nn import CrossEntropyLoss
from torchmetrics import JaccardIndex
from augment import LandCoverDataset, train_transform, val_transform
import rasterio
import numpy as np


# --------------------------------
# Config
# --------------------------------
BATCH_SIZE = 10
NUM_CLASSES = 10  # Adjust based on your classes
DEVICE = "cuda"

# 1. Load labels and determine NUM_CLASSES
with rasterio.open("../truth/landcover_labels.tif") as src:
    labels = src.read(1)
    unique_classes = np.unique(labels)
    print(unique_classes)
    NUM_CLASSES = int(np.max(unique_classes) + 1)  # e.g., 15
    print(f"Number of Classes: {NUM_CLASSES}")

# --------------------------------
# Dataset & Loaders
# --------------------------------
train_dataset = LandCoverDataset(
    image_dir="../data/train/images",
    label_dir="../data/train/labels",
    transform=train_transform,
)
val_dataset = LandCoverDataset(
    image_dir="../data/val/images",
    label_dir="../data/val/labels",
    transform=val_transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
)

# --------------------------------
# Model & Loss
# --------------------------------
model = smp.Unet(
    encoder_name="resnet18",  # Encoder with 4 blocks
    encoder_depth=4,  # Must match decoder_channels length
    decoder_channels=[256, 128, 64, 32],  # 4 values for 4 blocks
    in_channels=6,  # Landsat bands (B2–B7)
    classes=int(NUM_CLASSES),  # Number of land cover classes
).to(DEVICE)
model = model.float()

counts = np.zeros(NUM_CLASSES, dtype=np.float32)
unique_train, counts_train = np.unique(labels, return_counts=True)
for cls, cnt in zip(unique_train, counts_train):
    if cls < NUM_CLASSES:  # Ensure valid class indices
        counts[cls] = cnt
epsilon = 1e-3  # Larger smoothing factor
frequencies = (counts + epsilon) / (counts.sum() + epsilon)
class_weights = torch.tensor(np.log(1.0 / frequencies), dtype=torch.float32).to(
    DEVICE
)  # Log scaling

criterion = CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# --------------------------------
# Training Loop
# --------------------------------
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    jaccard = JaccardIndex(task="multiclass", num_classes=NUM_CLASSES).to(DEVICE)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            jaccard.update(preds, labels)
    return total_loss / len(loader), jaccard.compute()


# --------------------------------
# Run Training
# --------------------------------
NUM_EPOCHS = 1
best_iou = 0.0

print(f"CUDA Available: {torch.cuda.is_available()}")  # Should be True
print(f"GPU Name: {torch.cuda.get_device_name(0)}")  # Should show "Quadro RTX 3000"

for images, labels in train_loader:
    print("Images shape:", images.shape)  # Should be (B, 6, 256, 256)
    print("Labels shape:", labels.shape)  # Should be (B, 256, 256)
    break

for epoch in range(NUM_EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_iou = validate(model, val_loader, criterion)

    # Save best model logic...
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    print(
        f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}"
    )

# Save final model
torch.save(model, "../models/unet_landcover_final.pth")
