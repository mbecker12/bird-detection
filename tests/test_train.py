from cv2 import cv2
from glob import glob
from dml_project.util import *
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

sys.path.append(os.getcwd() + "/src/vision")
sys.path.append(os.getcwd() + "/src/vision/references/detection")
from types import ModuleType, FunctionType

from vision.references.detection.engine import train_one_epoch, evaluate
from torch.optim import Adam
import sys
import os
import pytest


def test_eval():
    try:
        _, faster_rcnn_model = define_model()
        val_dataloader = setup_dataloader(
            mode="val", batch_size=4, num_jpg=5, num_png=5
        )
        device = torch.device("cpu")
        faster_rcnn_model.eval()

        with torch.no_grad():
            coco_evaluator = evaluate(faster_rcnn_model, val_dataloader, device)
            # print(f"{coco_evaluator=}")
    except AssertionError as cuda_assert_err:
        print(cuda_assert_err)


def test_eval_show_images():
    _, faster_rcnn_model = define_model()
    val_dataloader = setup_dataloader(mode="val", batch_size=4, num_jpg=5, num_png=5)
    faster_rcnn_model.eval()

    with torch.no_grad():
        images, targets = next(iter(val_dataloader))
        outputs = faster_rcnn_model(images)

        for j, outp in enumerate(outputs):
            plot_img_and_boxes(
                None, images[j], normalize_boxes(outp["boxes"], images[j].shape)
            )
            # plt.show()


def test_train():
    _, faster_rcnn_model = define_model()
    train_dataloader = setup_dataloader(
        mode="train", batch_size=4, num_jpg=5, num_png=5
    )

    optimizer = Adam(faster_rcnn_model.parameters(), lr=0.0001)
    device = torch.device("cpu")
    print_freq = 1
    epoch = 0

    metric_logger = train_one_epoch(
        faster_rcnn_model, optimizer, train_dataloader, device, epoch, print_freq
    )

    # print(f"{metric_logger=}")


# def test_eval():
#     _, faster_rcnn_model = define_model()
#     val_dataloader = setup_dataloader(mode="val", batch_size=4, num_jpg=5, num_png=5)

#     optimizer = Adam(faster_rcnn_model.parameters(), lr=0.0001)
#     device = torch.device("cpu")
#     print_freq = 1
#     epoch = 0

#     faster_rcnn_model.eval()
#     with torch.no_grad():
#         images, targets = next(iter(val_dataloader))
#         outputs = faster_rcnn_model(images)

#         for j, outp in enumerate(outputs):
#             print(f"{outp.keys()=}")
#             plot_img_and_boxes(
#                 None, images[j], normalize_boxes(outp["boxes"], images[j].shape)
#             )
#             # plt.show()


# def test_eval_with_loaded_model():
#     _, faster_rcnn_model = define_model()
#     faster_rcnn_model.load_state_dict(torch.load("second_model"))
#     val_dataloader = setup_dataloader(mode="val", batch_size=6)

#     faster_rcnn_model.eval()
#     with torch.no_grad():
#         images, targets = next(iter(val_dataloader))
#         outputs = faster_rcnn_model(images)

#         for j, outp in enumerate(outputs):
#             plot_img_and_boxes(
#                 None, images[j], normalize_boxes(outp["boxes"], images[j].shape)
#             )
#             # plt.show()
