"""
NOTE: It has been shown that using the example below with the cv2 version
is more suitable, easier, and faster to use with torch tensors.
Hence, this module is not used but retained for benchmarking.

See augmentation.py for the preferred version.

Use albumenations (https://github.com/albumentations-team/albumentations) for image augmentations.
Based on examples provided by the package to use it together with torch/torchvision
(https://github.com/albumentations-team/albumentations_examples/blob/master/notebooks/migrating_from_torchvision_to_albumentations.ipynb)
"""

import numpy as np
import torch
import torchvision
from PIL import Image
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as a
from albumentations.pytorch import ToTensorV2
from time import time

DATA_PATH = "./data/"

class AlbumentationsDatasetPil(Dataset):
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

        image = Image.open(file_path)

        if self.transform:
            # Convert PIL image to numpy array
            image_np = np.array(image)
            # Apply transformations
            augmented = self.transform(image=image_np)
            if augmented["image"].dtype == np.float32:
                augmented["image"] *= 255
                augmented["image"] = augmented["image"].astype(np.uint8)
            # Convert numpy array to PIL Image
            image = Image.fromarray(augmented["image"])
            # image = transforms.ToTensor(image)
        return image, label


if __name__ == "__main__":
    start_pil = time()
    albumentations_pil_transform = a.Compose(
        [
            a.Resize(256, 256),
            a.HorizontalFlip(),
            a.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            # transforms.ToTensor()
            # ToTensorV2()
        ]
    )

    # Note that this dataset will output PIL images and not numpy arrays nor PyTorch tensors
    albumentations_pil_dataset = AlbumentationsDatasetPil(
        file_paths=glob(DATA_PATH + "*.jpg")[:50],
        labels=range(len(glob(DATA_PATH + "*.jpg")[:50])),
        transform=albumentations_pil_transform,
    )

    for x, y in albumentations_pil_dataset:
        print(type(x), type(y))
        print(min(x), max(x))
        pass

    end_pil = time()
    print(f"Time taken with PIL images: {end_pil - start_pil}")
