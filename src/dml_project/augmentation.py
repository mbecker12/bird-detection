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


def albumentations_transform(mode, normalize=True):
    """
    Creates a transform object to augment existing data. Applies the following augmentations:

        Training mode:
            * Shift, scale, rotate
            * Resize
            * Color jitter
            * Gaussian noise
            * Normalization (Optional)

        Validation mode:
            * Resize
            * Normalization (Optional)
    """
    if mode == "train":
        if normalize:
            augmentation = a.Compose(
                [
                    at.ShiftScaleRotate(
                        shift_limit=0.1,
                        scale_limit=(-0.1, 0.25),
                        rotate_limit=15,
                        p=0.5,
                    ),
                    a.Resize(MIN_HEIGHT, MIN_WIDTH),
                    at.ColorJitter(
                        brightness=0.08, contrast=0.06, saturation=0.06, hue=0.07, p=0.5
                    ),
                    at.GaussNoise(var_limit=0.1, p=0.5),
                    a.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                    ToTensorV2(),
                ],
                bbox_params=a.BboxParams(format="yolo", label_fields=["category_ids"]),
            )
        else:
            augmentation = a.Compose(
                [
                    at.ShiftScaleRotate(
                        shift_limit=0.1,
                        scale_limit=(-0.1, 0.25),
                        rotate_limit=15,
                        p=0.5,
                    ),
                    a.Resize(MIN_HEIGHT, MIN_WIDTH),
                    at.ColorJitter(
                        brightness=0.08, contrast=0.06, saturation=0.06, hue=0.07, p=0.5
                    ),
                    at.GaussNoise(var_limit=0.1, p=0.5),
                    a.ToFloat(),
                    ToTensorV2(),
                ],
                bbox_params=a.BboxParams(format="yolo", label_fields=["category_ids"]),
            )
    elif mode == "val":
        if normalize:
            augmentation = a.Compose(
                [
                    a.Resize(MIN_HEIGHT, MIN_WIDTH),
                    a.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                    ToTensorV2(),
                ],
                bbox_params=a.BboxParams(
                    format="yolo", label_fields=["category_ids"], min_visibility=0.2
                ),
            )
        else:
            augmentation = a.Compose(
                [
                    a.Resize(MIN_HEIGHT, MIN_WIDTH),
                    a.ToFloat(),
                    ToTensorV2(),
                ],
                bbox_params=a.BboxParams(
                    format="yolo", label_fields=["category_ids"], min_visibility=0.2
                ),
            )
    else:
        raise Exception(
            f"Error! Mode {mode} not supported for choosing an augmentation pipeline. Must be either 'train' or 'val'."
        )

    return augmentation


class AlbumentationsDatasetCV2(Dataset):
    """
    Custom dataset class which incorporates the data augmentation classes above.
    __init__ and __len__ functions are the same as in TorchvisionDataset
    """

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
                image=image,
                bboxes=target["boxes"],
                category_ids=target["labels"],
                image_id=target["image_id"],
                iscrowd=target["iscrowd"],
                area=target["area"],
            )

            targets = {
                "boxes": torch.as_tensor(augmented["bboxes"]).reshape(-1, 4),
                "labels": torch.as_tensor(augmented["category_ids"]),
                "image_id": torch.as_tensor(augmented["image_id"]),
                "area": torch.as_tensor(augmented["area"]),
                "iscrowd": torch.as_tensor(augmented["iscrowd"]),
            }

            return augmented["image"], targets

        return image, target


if __name__ == "__main__":
    """
    Main function to test the data augmentation on a 
    set of images and draw the transformed images
    """
    images = load_images(DATA_PATH, num_jpg=2, num_png=5)
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
        bbox_params=a.BboxParams(
            format="yolo",
            label_fields=["category_ids"],
        ),
    )

    albumentations_dataset = AlbumentationsDatasetCV2(
        file_paths=load_images(DATA_PATH),
        transform=albumentations_transform,
    )

    for x, boxes, cid in albumentations_dataset:
        print(type(x), x.dtype, len(boxes), len(cid))

    end_cv2 = time()
    print(f"Time taken with CV2 images: {end_cv2 - start_cv2}")
