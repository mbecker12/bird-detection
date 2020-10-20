"""
Test the application of a full data augmentation pipeline.
Test the loading, resizing, transformation, augmentation of images;
and possibly changing its datatype to one that is needed by pytorch.
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


def test_augmentation_pipeline_only_img():
    """
    Check that a predefined augmentation pipeline behaves as expected on few example images (without bboxes).
    Assure that datatypes stay as expected and is in a pytorch-usable format at the end.
    """
    image_paths = load_images(DATA_PATH, num_jpg=3, num_png=3)
    resize_width = 512
    resize_height = 512
    albumentations_transform = a.Compose(
        [
            at.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=(-0.1, 0.5), rotate_limit=15, p=1.0
            ),
            a.Resize(resize_height, resize_width),
            at.ColorJitter(
                brightness=0.1, contrast=0.07, saturation=0.07, hue=0.07, p=1.0
            ),
            at.GaussNoise(var_limit=0.1, p=1.0),
            a.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
            ToTensorV2(),
        ]
    )

    dataset = AlbumentationsDatasetCV2(
        file_paths=image_paths,
        transform=albumentations_transform,
    )

    for i, (img, targets) in enumerate(dataset):
        assert -2 <= img.min() <= 0
        assert 0 <= img.max() <= 2

        # plot_img_and_boxes(image_paths[i])
        # plot_img_and_boxes(None, img, boxes)
        # plt.show()


def test_augmentation_pipeline_with_bbox():
    """
    Check that a predefined augmentation pipeline behaves as expected on few example images and their bounding boxes.
    Assure that datatypes stay as expected and is in a pytorch-usable format at the end.
    Also assure that the bounding boxes still correspond to their intended objects.
    """
    image_paths = load_images(DATA_PATH, num_jpg=3, num_png=3)
    resize_width = 512
    resize_height = 512
    albumentations_transform = a.Compose(
        [
            at.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=(-0.1, 0.5), rotate_limit=15, p=1.0
            ),
            a.Resize(resize_height, resize_width),
            at.ColorJitter(
                brightness=0.1, contrast=0.07, saturation=0.07, hue=0.07, p=1.0
            ),
            at.GaussNoise(var_limit=0.1, p=1.0),
            a.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
            ToTensorV2(),
        ],
        bbox_params=a.BboxParams(format="yolo", label_fields=["category_ids"]),
    )

    dataset = AlbumentationsDatasetCV2(
        file_paths=image_paths,
        transform=albumentations_transform,
    )

    for i, (img, targets) in enumerate(dataset):
        
        boxes = targets["boxes"] 
        class_idx = targets["labels"]
        assert -2 <= img.min() <= 0
        assert 0 <= img.max() <= 2

        for box in boxes:
            assert 0 <= box[0] <= 1
            assert 0 <= box[1] <= 1
            assert 0 <= box[2] <= 1
            assert 0 <= box[3] <= 1
        # print(f"{boxes=}")
        target = load_bbox_file(image_paths[i])

        box_file_path = image_paths[i].split(".")[-2] + ".txt"

        # gt_boxes = target["boxes"]
        # print(f"{gt_boxes=}")
        plot_img_and_boxes(image_paths[i])
        plot_img_and_boxes(None, img, boxes)
        # plt.show()
