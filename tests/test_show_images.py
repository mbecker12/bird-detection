"""
Test utility functions for plotting.
"""

from cv2 import cv2
from glob import glob
from dml_project.util import show_img, load_images, load_bbox_file, plot_img_and_boxes
from dml_project.const import *
from torchvision import transforms
import albumentations as a
from albumentations.pytorch import ToTensorV2
from dml_project.augmentation import AlbumentationsDatasetCV2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch


def test_show_single_img():
    images = load_images(DATA_PATH, num_jpg=1, num_png=1)

    for img in images:
        show_img(img)


def test_show_image_and_boxes():
    image_paths = load_images(DATA_PATH, num_jpg=1, num_png=1)

    albumentations_transform = a.Compose(
        [
            a.Resize(512, 512),
            ToTensorV2(),
        ],
        bbox_params=a.BboxParams(format="yolo", label_fields=["category_ids"]),
    )

    dataset = AlbumentationsDatasetCV2(
        file_paths=image_paths,
        transform=albumentations_transform,
    )

    for i, img in enumerate(image_paths):

        dat_img, boxes, cid = dataset[i]

        _dat_img = dat_img.permute(1, 2, 0)

        plot_img_and_boxes(img)
        plot_img_and_boxes(None, _dat_img, boxes)
