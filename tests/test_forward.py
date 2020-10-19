"""
Test the forward function of a defined network
to make sure that the images and bounding boxes can be used
by the network and that the network returns bounding boxes
as predictions for given images.
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
from dml_project.model import define_model, setup_dataloader
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

def test_forward():
    """
    Based on the defined dataset, dataloader, and augmentation pipeline,
    make sure that all data points plus bounding boxes can be used by the pretrained model
    and that all shapes and datatypes fit so that the model can give a sensible output.
    """

    model, faster_rcnn_model = define_model()
    data_loader = setup_dataloader("train", batch_size=16, num_workers=0, shuffle=False)
    print()
    # for training
    images, bboxes, cids = next(iter(data_loader))
    # print(images, bboxes, cids)
    print(f"{len(images)=}")
    # images, bboxes, cids = next(data_loader)
    # for images, bboxes, cids in data_loader:
        # print(images, bboxes, cids)
    images = list(image for image in images)
    n_samples = len(images)

    bboxes_ = []
    for box in bboxes:
        box_ = [b.item() for b in box]
        bboxes_.append(box_)

    print()
    print(f"{bboxes=}")
    print(f"{bboxes_=}")
    print(f"{cids=}")

    targets = [{"boxes": bboxes_[i], "labels": cids[i].item()} for i in range(n_samples)]

    output = model(images, targets)
    print(output)

    # for inference
    model.eval()

    x = [torch.rand(3, 1000, 1000), torch.rand(3, 1200, 900)]

    predictions = model(x)
    print(predictions)