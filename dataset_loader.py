from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class TumorCoreDataset(Dataset):
    def __init__(self, subject_dirs, transform=None):
        self.image_mask_pairs = []
        self.transform = transform

        for subject_path in subject_dirs:
            t1ce_dir = os.path.join(subject_path, 't1ce')
            seg_dir = os.path.join(subject_path, 'seg')

            if not os.path.exists(t1ce_dir) or not os.path.exists(seg_dir):
                continue

            for fname in os.listdir(t1ce_dir):
                img_file = os.path.join(t1ce_dir, fname)
                mask_file = os.path.join(seg_dir, fname)
                if os.path.exists(mask_file):
                    self.image_mask_pairs.append((img_file, mask_file))

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.image_mask_pairs[idx]
        img = Image.open(img_path).convert("L").resize((256, 256))
        mask = Image.open(mask_path).convert("L").resize((256, 256))

        img = transforms.ToTensor()(img)
        mask = transforms.ToTensor()(mask)

        return img, mask
