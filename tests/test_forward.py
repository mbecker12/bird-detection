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