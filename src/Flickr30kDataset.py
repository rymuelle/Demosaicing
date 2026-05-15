import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch

from PIL import Image

def apply_gamma(image, gamma):
    lut = [pow(i / 255.0, gamma) * 255 for i in range(256)]
    lut = lut * len(image.getbands())
    return image.point(lut)


class Flickr30kDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_file (string): Path to the csv file with captions.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = pd.read_csv(csv_file, delimiter=',')
        self.root_dir = root_dir
        self.transform = transform
        
        # Clean up column names (Kaggle versions often have leading spaces)
        self.df.columns = [col.strip() for col in self.df.columns]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        caption = self.df.iloc[idx, 1]
        
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        # image = apply_gamma(image, 2.2)
        if self.transform:
            image = self.transform(image)

        return image, caption