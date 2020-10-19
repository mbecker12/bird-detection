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
from albumentations.augmentations import transforms as at
from albumentations.pytorch import ToTensorV2
from time import time
from dml_project.util import load_images, load_bbox_file
from dml_project.const import *


def albumentations_transform(mode):
    if mode == "train":
        augmentation = a.Compose(
            [
                at.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=(-0.1, 0.5), rotate_limit=15, p=0.5
                ),
                a.Resize(MIN_HEIGHT, MIN_WIDTH),
                at.ColorJitter(
                    brightness=0.1, contrast=0.07, saturation=0.07, hue=0.07, p=0.5
                ),
                at.GaussNoise(var_limit=0.1, p=0.5),
                a.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
                ToTensorV2(),
            ],
            bbox_params=a.BboxParams(format="yolo", label_fields=["category_ids"]),
        )
    elif mode == "val":
        augmentation = a.Compose(
            [
                a.Resize(MIN_HEIGHT, MIN_WIDTH),
                a.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
                ToTensorV2(),
            ],
            bbox_params=a.BboxParams(format="yolo", label_fields=["category_ids"]),
        )
    else:
        raise Exception(
            f"Error! Mode {mode} not supported for choosing an augmentation pipeline. Must be either 'train' or 'val'."
        )

    return augmentation


class AlbumentationsDatasetCV2(Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""

    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths

        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        target = load_bbox_file(file_path)
        # Read an image with OpenCV
        image = cv2.imread(file_path)

        # By default OpenCV uses BGR color space for color images,
        # so we need to convert the image to RGB color space.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(
                image=image, bboxes=target["boxes"], category_ids=target["labels"]
            )
            # augmented, target = self.transform(image=image, target=target)
            # return augmented, target["boxes"], target["category_ids"]
            return augmented["image"], augmented["bboxes"], augmented["category_ids"]

        return image, target["boxes"], target["labels"]


if __name__ == "__main__":
    images = load_images(DATA_PATH, num_jpg=2, num_png=5)
    # import matplotlib.pyplot as plt
    # for img in images:
    #     cv2_img = cv2.imread(img)
    #     normal_image = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    #     plt.imshow(normal_image)
    #     plt.show()
    #     # cv2.imshow("Some Image", cv2_img)

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
        ],
        bbox_params=a.BboxParams(format="yolo", label_fields=["category_ids"]),
    )

    albumentations_dataset = AlbumentationsDatasetCV2(
        file_paths=load_images(DATA_PATH),
        transform=albumentations_transform,
    )

    for x, boxes, cid in albumentations_dataset:
        print(type(x), x.dtype, len(boxes), len(cid))

    end_cv2 = time()
    print(f"Time taken with CV2 images: {end_cv2 - start_cv2}")
