import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import glob
from EUnet import UNetSimplified
from dataset_loader import TumorCoreDataset
from torchvision import transforms

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Resize transform
resize_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Load training subjects
print("Loading training data...")
train_subjects = glob.glob('./dataset/BraTS2021_Split/train/*')
print(f"Found {len(train_subjects)} subjects.")

train_dataset = TumorCoreDataset(train_subjects, transform=resize_transform)
print(f"Total slices in dataset: {len(train_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Model setup
model = UNetSimplified(in_channels=1, out_channels=1).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

print("Starting training...\n")

# Training loop
for epoch in range(10):
    model.train()
    total_loss = 0.0

    for batch_idx, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)

        preds = model(images)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
            print(f"Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} completed. Total Loss: {total_loss:.4f}\n")

# Save the model
torch.save(model.state_dict(), 'eunet_tumor_core.pth')
print("Model saved as 'eunet_tumor_core.pth'")

