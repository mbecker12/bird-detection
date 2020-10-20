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
import sys

from types import ModuleType, FunctionType
from gc import get_referents

# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST = type, ModuleType, FunctionType


def getsize(obj):
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError("getsize() does not take argument of type: " + str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size


def test_forward_eval():
    # for inference
    _, faster_rcnn_model = define_model()
    faster_rcnn_model.eval()

    x = [torch.rand(3, 1000, 1000), torch.rand(3, 1200, 900)]

    predictions = faster_rcnn_model(x)


def test_forward_train():
    """
    Based on the defined dataset, dataloader, and augmentation pipeline,
    make sure that all data points plus bounding boxes can be used by the pretrained model
    and that all shapes and datatypes fit so that the model can give a sensible output.
    """

    _, faster_rcnn_model = define_model()
    data_loader = setup_dataloader("train", batch_size=4, num_workers=0, shuffle=False)
    # for training
    images, targets = next(iter(data_loader))

    # images = list(image for image in images)
    n_samples = len(images)

    # targets = [{"boxes": bboxes[i], "labels": cids[i]} for i in range(n_samples)]
    output = faster_rcnn_model(images, targets)
