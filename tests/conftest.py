# -*- coding: utf-8 -*-
"""
    Dummy conftest.py for dml_project.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    https://pytest.org/latest/plugins.html
"""

import pytest
import numpy as np
import torch
import torchvision
from PIL import Image
from cv2 import cv2
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as a
from albumentations.pytorch import ToTensorV2
from time import time
