import torch
from torch.utils.data import DataLoader, Subset
from torch import nn, optim
import glob
import random
from torchvision import transforms
import matplotlib.pyplot as plt
from EUnet import UNetSimplified
from dataset_loader import TumorCoreDataset
import os

def visualize_prediction(image_tensor, mask_tensor, pred_tensor, save_path=None):
    image = image_tensor.squeeze().cpu().numpy()
    mask = mask_tensor.squeeze().cpu().numpy()
    pred = pred_tensor.squeeze().detach().cpu().numpy()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Input (T1ce)')
    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title('Ground Truth')
    axs[2].imshow(pred > 0.5, cmap='gray')  # binary prediction
    axs[2].set_title('Prediction')

    for ax in axs:
        ax.axis('off')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

resize_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

print("Loading training data...")
train_subjects = glob.glob('./dataset/BraTS2021_Split/train/*')
print(f"Found {len(train_subjects)} subject folders.")

full_dataset = TumorCoreDataset(train_subjects, transform=resize_transform)
print(f"Total available slices: {len(full_dataset)}")

subset_size = 5000
subset_indices = random.sample(range(len(full_dataset)), subset_size)
train_dataset = Subset(full_dataset, subset_indices)
print(f"Using {len(train_dataset)} randomly sampled slices for training.")

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

model = UNetSimplified(in_channels=1, out_channels=1).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

print("Starting training...\n")

for epoch in range(3):
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
        
        if (batch_idx + 1) % 100 == 0:
            save_path = f'output_visuals/epoch{epoch+1}_batch{batch_idx+1}.png'
            visualize_prediction(images[0], masks[0], preds[0], save_path)          

    print(f"Epoch {epoch+1} completed. Total Loss: {total_loss:.4f}\n")

torch.save(model.state_dict(), 'eunet_tumor_core.pth')
print("Model saved as 'eunet_tumor_core.pth'")

