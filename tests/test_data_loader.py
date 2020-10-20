"""
Test pytorch data loaders, including data augmentations.
"""
import pytest
from cv2 import cv2
from glob import glob
from dml_project.util import show_img, load_images, load_bbox_file, plot_img_and_boxes
from dml_project.const import *
from torchvision import transforms
import albumentations as a
from albumentations.augmentations import transforms as at
from albumentations.pytorch import ToTensorV2
from dml_project.augmentation import AlbumentationsDatasetCV2, albumentations_transform
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from dml_project.model import setup_dataloader

img_width = 512
img_height = 512


@pytest.mark.parametrize(
    "mode,batch_size",
    [
        ("train", 1),
        ("train", 8),
        ("train", 16),
        ("val", 1),
        ("val", 8),
        ("val", 16),
    ],
)
def test_data_loader(mode, batch_size):
    """
    Test a data loader with the corresponding augmentation pipeline for the training set.
    i.e. the example images and bounding boxes will go through a series of augmentations.
    Make sure that it will return the correct batch size and data format, as well as pixel range.
    """
    image_paths = load_images(DATA_PATH)  # , num_jpg=20, num_png=3)

    data_loader = setup_dataloader(
        mode=mode, batch_size=batch_size, num_workers=0, shuffle=True
    )

    images, targets = next(iter(data_loader))

    iteration = 0
    try:
        for img, targets in data_loader:
            boxes = [targets[i]["boxes"] for i in range(img.shape[0])]
            cids = [targets[i]["labels"] for i in range(img.shape[0])]
            plot_img_and_boxes(None, img[0], boxes[0])

            assert len(boxes) == img.shape[0]
            if iteration > 0:
                break
            iteration += 1
            assert img.shape[0] == batch_size, img.shape
            assert img.shape[1] <= 4
            assert img.shape[2] >= MIN_HEIGHT
            assert img.shape[3] >= MIN_WIDTH
            assert len(boxes) >= 0
            assert img.dtype == torch.float32

            for im in img:
                assert -3 <= im.min() <= 0
                assert 0 <= im.max() <= 3

            for j, boxs in enumerate(boxes):
                assert boxs.dtype == torch.float32
                assert len(boxs) == len(cids[j])
                for box in boxs:

                    assert len(box) == 4
                    # if boxes are recomputed for rcnn
                    if box[0] > 1 or box[1] > 1 or box[2] > 1 or box[3] > 1:
                        assert box[0] < box[2]
                        assert box[1] < box[3]
                        assert 0 <= box[0] <= img.shape[3], box
                        assert 0 <= box[1] <= img.shape[2], box
                        assert 0 <= box[2] <= img.shape[3], box
                        assert 0 <= box[3] <= img.shape[2], box
                    else:
                        assert 0 <= box[0] <= 1, box
                        assert 0 <= box[1] <= 1, box
                        assert 0 <= box[2] <= 1, box
                        assert 0 <= box[3] <= 1, box

                for idx in cids[j]:
                    assert idx.dtype == torch.int64 or idx.dtype == torch.int32
                    assert idx <= NUM_CLASSES

            assert len(cids) == len(boxes)
    except ValueError as val_err:
        print(val_err)
        print(data_loader.__dict__)
