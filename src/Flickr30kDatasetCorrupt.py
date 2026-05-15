import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import random

def random_crop_to_numpy(img, crop_size, mode='center'):

    """

    Randomly crops a PIL image and returns a NumPy array.

    """

    width, height = img.size

    # Randomly select top-left corner
    left = random.randint(0, max(0, width - crop_size[0]))
    top = random.randint(0, max(0, height - crop_size[1]))
    if mode == "center":
        left = (width - crop_size[0]) // 2
        top = (height - crop_size[1]) // 2
    # Define crop area (left, top, right, bottom)
    right = left + crop_size[0]
    bottom = top + crop_size[1]

    # Crop and convert to numpy
    cropped_img = img.crop((left, top, right, bottom))
    return np.array(cropped_img)



def apply_gamma(image, gamma):
    lut = [pow(i / 255.0, gamma) * 255 for i in range(256)]
    lut = lut * len(image.getbands())
    return image.point(lut)


class Flickr30kDatasetCorrupt(Dataset):
    def __init__(self, root_dir, csv_file, corrupt, crop_size=(256, 256), transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_file (string): Path to the csv file with captions.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = pd.read_csv(csv_file, delimiter=',')
        self.root_dir = root_dir
        self.transform = transform
        self.corrupt = corrupt
        self.crop_size = crop_size
        
        self.df.columns = [col.strip() for col in self.df.columns]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        caption = self.df.iloc[idx, 1]

        img_path = os.path.join(self.root_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        image = random_crop_to_numpy(image, self.crop_size).astype(np.float32)

        corrupted = self.corrupt(image)
        if self.transform:
            image = self.transform(image)
            corrupted = self.transform(corrupted)

        return image, corrupted, caption