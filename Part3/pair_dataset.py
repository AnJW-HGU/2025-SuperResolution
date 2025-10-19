from torch.utils.data import Dataset
from PIL import Image
import os

class PairedDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform_lr=None, transform_hr=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.transform_lr = transform_lr
        self.transform_hr = transform_hr
        self.classes = sorted(os.listdir(lr_dir))
        self.samples = []
        self.targets = [] 

        for label, cls in enumerate(self.classes):
            lr_cls_dir = os.path.join(lr_dir, cls)
            # hr_cls_dir = os.path.join(hr_dir, cls)
            for img_name in os.listdir(lr_cls_dir):
                lr_path = os.path.join(lr_cls_dir, img_name)
                hr_path = os.path.join(hr_dir, img_name)
                if os.path.exists(hr_path):
                    self.samples.append((lr_path, hr_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        lr_path, hr_path, label = self.samples[idx]
        lr_img = Image.open(lr_path).convert('RGB')
        hr_img = Image.open(hr_path).convert('RGB')

        if self.transform_lr:
            lr_img = self.transform_lr(lr_img)

        if self.transform_hr:
            hr_img = self.transform_hr(hr_img)

        return lr_img, hr_img, label
