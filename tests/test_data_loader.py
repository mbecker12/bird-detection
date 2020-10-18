"""
Test pytorch data loaders, including data augmentations.
"""
from cv2 import cv2
from glob import glob
from dml_project.util import show_img, load_images, load_bbox_file, plot_img_and_boxes
from dml_project.const import *
from torchvision import transforms
import albumentations as a
from albumentations.augmentations import transforms as at
from albumentations.pytorch import ToTensorV2
from dml_project.augmentation import AlbumentationsDatasetCV2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

img_width = 512
img_height = 512

albumentations_transform = {
    'train': a.Compose(
            [
                at.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=(-0.1, 0.5), 
                    rotate_limit=15, p=0.5),
                a.Resize(img_height, img_width),
                at.ColorJitter(brightness=0.1, contrast=0.07, saturation=0.07, hue=0.07, p=0.5),
                at.GaussNoise(var_limit=0.1, p=0.5),
                a.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
                ToTensorV2(),
            ],
            bbox_params=a.BboxParams(format='yolo', label_fields=['category_ids'])
        ),
    'val': a.Compose(
            [
                a.Resize(img_height, img_width),
                a.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
                ToTensorV2(),
            ],
            bbox_params=a.BboxParams(format='yolo', label_fields=['category_ids'])
        ), 
}
    

def test_data_loader():
    """
    Test a data loader with the corresponding augmentation pipeline for the training set.
    i.e. the example images and bounding boxes will go through a series of augmentations.
    Make sure that it will return the correct batch size and data format, as well as pixel range.
    """
    for mode in ('train', 'val'):
        for batch_size in (1, 8, 16):
            image_paths = load_images(DATA_PATH, num_jpg=20, num_png=3)

            dataset = AlbumentationsDatasetCV2(
                file_paths=image_paths,
                transform=albumentations_transform[mode],
            )

            data_loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=0,
                shuffle=False
            )

            iteration = 0
            for img, boxes, cids in data_loader:
                if iteration > 0:
                    break
                iteration += 1
                assert img.shape[0] == batch_size, img.shape
                assert img.shape[1] <= 4
                assert img.shape[2] == img_height
                assert img.shape[3] == img_width

                for im in img:
                    assert -2 <= im.min() <= 0
                    assert 0 <= im.max() <= 2

                for j, box in enumerate(boxes):
                    box = [b.item() for b in box]
                    assert len(box) == 4
                    assert cids[j].item() <= 6 # N_CLASSES
                    assert 0 <= box[0] <= 1 
                    assert 0 <= box[1] <= 1 
                    assert 0 <= box[2] <= 1 
                    assert 0 <= box[3] <= 1 

                assert len(cids) == len(boxes)
