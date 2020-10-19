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
            # ("train", 16),
            ("val", 1),
            ("val", 8),
            # ("val", 16),
        ]
    )
def test_data_loader(mode, batch_size):
    """
    Test a data loader with the corresponding augmentation pipeline for the training set.
    i.e. the example images and bounding boxes will go through a series of augmentations.
    Make sure that it will return the correct batch size and data format, as well as pixel range.
    """
    image_paths = load_images(DATA_PATH, num_jpg=20, num_png=3)

    # dataset = AlbumentationsDatasetCV2(
    #     file_paths=image_paths,
    #     transform=albumentations_transform(mode),
    # )

    # data_loader = DataLoader(
    #     dataset=dataset,
    #     batch_size=batch_size,
    #     num_workers=0,
    #     shuffle=False
    # )

    data_loader = setup_dataloader(mode=mode, batch_size=batch_size, num_workers=0, shuffle=True)


    iteration = 0
    for next_iter in data_loader:
        for item in next_iter:
            print(f"{len(item)=}")
        if iteration > 0:
            break
        iteration += 1
        # print(f"{next_iter=}")
        # print(f"{type(next_iter)=}")
        # print(f"{len(next_iter)=}")

    iteration = 0
    for img, boxes, cids in data_loader:

        plot_img_and_boxes(None, img[0], boxes)
        plt.show()
        print(f"{img.shape=}")
        # print(f"{boxes=}")
        print(f"{len(boxes)=}")
        assert len(boxes) == img.shape[0]
        if iteration > 0:
            break
        iteration += 1
        assert img.shape[0] == batch_size, img.shape
        assert img.shape[1] <= 4
        assert img.shape[2] >= MIN_HEIGHT
        assert img.shape[3] >= MIN_WIDTH
        assert len(boxes) >= 0

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
