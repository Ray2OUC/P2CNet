import torch
import os
import cv2
import kornia
from glob import glob
from torch.utils.data import Dataset


class TestDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.uw_images = glob(os.path.join(self.data_path, '*.' + 'jpg')) + glob(os.path.join(self.data_path, '*.' + 'JPEG')) + glob(os.path.join(self.data_path, '*.' + 'png'))
        self.transform = self.test_transform

    def test_transform(self, img):
        img = cv2.imread(img)
        img = cv2.resize(img, (256, 256))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb = torch.tensor(rgb.astype(float).transpose(2, 0, 1), dtype=torch.float) / 255.  # 3 h w
        lab = kornia.color.rgb_to_lab(rgb)  # l: 0~100; a,b: -127~127
        l = lab[:1, :, :] / 100.  # 0~1
        ab = lab[1:, :, :] / 127.  # -1~1
        lab = torch.cat([l, ab], dim=0)
        return lab
    
    def __getitem__(self, index):
        uw_img = self.transform(self.uw_images[index])
        name = os.path.basename(self.uw_images[index])
        return uw_img, name

    def __len__(self):
        return len(self.uw_images)
