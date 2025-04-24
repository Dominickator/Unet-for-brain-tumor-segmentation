import torch
from torch.utils.data import DataLoader, Subset
import glob
import random
import os
import matplotlib.pyplot as plt
from torchvision import transforms

from EUnet import UNetSimplified
from dataset_loader import TumorCoreDataset
from utils import dice_score  # assumes pred is already sigmoid-ed

def visualize_prediction(image_tensor, mask_tensor, pred_tensor, save_path=None):
    image = image_tensor.squeeze().cpu().numpy()
    mask = mask_tensor.squeeze().cpu().numpy()
    pred_prob = pred_tensor.squeeze().detach().cpu().numpy()  
    pred_bin = (pred_prob > 0.5).astype(float)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Input (T1ce)')

    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title('Ground Truth')

    axs[2].imshow(pred_prob, cmap='hot')
    axs[2].set_title('Prediction Prob')
    fig.colorbar(axs[2].imshow(pred_prob, cmap='hot'), ax=axs[2], fraction=0.046, pad=0.04)

    axs[3].imshow(pred_bin, cmap='gray')
    axs[3].set_title('Prediction Binary')

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

# Load dataset
test_subjects = glob.glob('./dataset/BraTS2021_Split/test/*')
test_dataset = TumorCoreDataset(test_subjects, transform=resize_transform)

# Randomly sample 2143 slices
subset_size = min(2143, len(test_dataset))
random.seed(42)
random_indices = random.sample(range(len(test_dataset)), subset_size)

test_subset = Subset(test_dataset, random_indices)
test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)

# Load model
model = UNetSimplified(in_channels=1, out_channels=1).to(device)
model.load_state_dict(torch.load(
    'eunet_tumor_core.pth',
    map_location=device))
model.eval()

# Create output dirs
os.makedirs("prediction_output/visuals", exist_ok=True)
os.makedirs("prediction_output/masks", exist_ok=True)

total_dice = 0.0

with torch.no_grad():
    for i, (img, mask) in enumerate(test_loader):
        img, mask = img.to(device), mask.to(device)

        output = model(img)  # already sigmoid-ed inside model
        dice = dice_score(output, mask)
        total_dice += dice.item()

        print(f"Sample {i+1}: Dice = {dice.item():.4f} | Max prob = {output.max().item():.3f}")

        if i < 5:
            save_path = f"prediction_output/visuals/sample_{i+1}_full.png"
            visualize_prediction(img, mask, output, save_path)

            pred_bin = (output.squeeze().cpu() > 0.5).float()
            torch.save(pred_bin, f"prediction_output/masks/sample_{i+1}_pred_mask.pt")

avg_dice = total_dice / len(test_loader)
print(f"\nAverage Dice Score on Random 2143 Slices: {avg_dice:.4f}")
