import torch
from torch.utils.data import DataLoader
import glob
from torchvision import transforms
from EUnet import UNetSimplified
from dataset_loader import TumorCoreDataset
from utils import dice_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resize_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

test_subjects = glob.glob('./dataset/BraTS2021_Split/test/*')
test_dataset = TumorCoreDataset(test_subjects, transform=resize_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = UNetSimplified(in_channels=1, out_channels=1).to(device)
model.load_state_dict(torch.load('eunet_tumor_core.pth', map_location=device))
model.eval()

total_dice = 0.0

with torch.no_grad():
    for i, (img, mask) in enumerate(test_loader):
        img, mask = img.to(device), mask.to(device)
        output = model(img)
        dice = dice_score(output, mask)
        total_dice += dice.item()
        print(f"Sample {i+1}: Dice Score = {dice.item():.4f}")

avg_dice = total_dice / len(test_loader)
print(f"\nAverage Dice Score on Test Set: {avg_dice:.4f}")
