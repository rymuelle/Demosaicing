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

import cv2
import numpy as np

def add_luma_noise(img, noise_intensity=0.1):
    img *= 255
    lab = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    noise = np.random.normal(loc=0.0, scale=noise_intensity, size=l_channel.shape)
    # kernel_size=3
    # noise = cv2.GaussianBlur(noise, (kernel_size, kernel_size), 0)
    # noise += np.random.normal(loc=0.0, scale=noise_intensity, size=l_channel.shape)*.5
    # print(noise.std())
    noisy_l = l_channel.astype(np.float32) + noise * 255
    noisy_l = np.clip(noisy_l, 0, 255).astype(np.uint8)
    lab[:, :, 0] = noisy_l
    final_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype(np.float32)
    return final_img / 255


def scale_noise(img, noise_intensity=0.1):
    h, w, c = img.shape
    noise = 1 + np.random.normal(loc=0.0, scale=noise_intensity, size=(h, w, 1))*noise_intensity
    return img * noise


class ImageDatasetCorrupt(Dataset):
    def __init__(self, root_dir, corrupt, crop_size=(256, 256), noise=0, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.files = [f for f in os.listdir(root_dir) if f[-4:] == ".jpg"]
        self.corrupt = corrupt
        self.crop_size = crop_size
        self.noise = noise

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]

        img_path = os.path.join(self.root_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        image = random_crop_to_numpy(image, self.crop_size).astype(np.float32) / 255
        if self.noise > 0:
            h, w, c = image.shape
            sigma = random.random() * self.noise
            image = add_luma_noise(image, sigma)

        corrupted = self.corrupt(image)
        image = torch.tensor(image).permute(2, 0, 1)
        corrupted = torch.tensor(corrupted).permute(2, 0, 1)

        return image, corrupted