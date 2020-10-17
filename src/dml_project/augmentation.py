"""
Use albumenations (https://github.com/albumentations-team/albumentations) for image augmentations.
Based on examples provided by the package to use it together with torch/torchvision
(https://github.com/albumentations-team/albumentations_examples/blob/master/notebooks/migrating_from_torchvision_to_albumentations.ipynb)
"""

import numpy as np
import torch
import torchvision
from PIL import Image
from cv2 import cv2
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as a
from albumentations.pytorch import ToTensorV2
from time import time


DATA_PATH = "./data/"


class AlbumentationsDatasetCV2(Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""

    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        file_path = self.file_paths[idx]

        # Read an image with OpenCV
        image = cv2.imread(file_path)

        # By default OpenCV uses BGR color space for color images,
        # so we need to convert the image to RGB color space.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return image, label


if __name__ == "__main__":
    start_cv2 = time()
    albumentations_transform = a.Compose(
        [
            a.Resize(256, 256),
            a.HorizontalFlip(),
            a.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )

    albumentations_dataset = AlbumentationsDatasetCV2(
        file_paths=glob(DATA_PATH + "*.jpg")[:50],
        labels=range(len(glob(DATA_PATH + "*.jpg")[:50])),
        transform=albumentations_transform,
    )

    for x, y in albumentations_dataset:
        print(type(x), x.dtype, type(y))

    end_cv2 = time()
    print(f"Time taken with CV2 images: {end_cv2 - start_cv2}")
