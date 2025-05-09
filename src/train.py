import rasterio
import torch
from torch.utils.data import DataLoader
from config import *
from src.model import create_unet
from src.data_processing import LandcoverDataset

class LandcoverDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_files = list(img_dir.glob("*.tif"))
        self.mask_files = [mask_dir / f.name for f in self.img_files]
        self.transform = transform
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        with rasterio.open(self.img_files[idx]) as img:
            image = img.read()[:3]  # Use RGB bands only
        with rasterio.open(self.mask_files[idx]) as msk:
            mask = msk.read(1)
            
        if self.transform:
            augmented = self.transform(image=image.transpose(1,2,0), mask=mask)
            image = augmented["image"].transpose(2,0,1)
            mask = augmented["mask"]
            
        return torch.tensor(image), torch.tensor(mask)

def train():
    model = create_unet().to(DEVICE)
    dataset = LandcoverDataset(PROCESSED_DIR/"images", PROCESSED_DIR/"masks", train_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        for images, masks in loader:
            # Training loop remains same
            ...

if __name__ == "__main__":
    train()