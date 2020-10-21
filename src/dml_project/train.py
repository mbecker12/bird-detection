import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from dml_project.augmentation import AlbumentationsDatasetCV2, albumentations_transform
from dml_project.const import NUM_CHANNELS, NUM_CLASSES
from glob import glob
from dml_project.const import *
from dml_project.util import *
from dml_project.model import define_model, setup_dataloader
from torch.utils.data import DataLoader
import torch
from torch.optim import Adam
import sys
import os

# print(f"{os.getcwd()=}")
sys.path.append(os.getcwd() + "/src/vision/references/detection")
# print(f"{sys.path=}")
from vision.references.detection.engine import train_one_epoch, evaluate


if __name__ == "__main__":
    if sys.argv[1] == "train":
        _, faster_rcnn_model = define_model()
        train_dataloader = setup_dataloader(mode="train", batch_size=4)
        val_dataloader = setup_dataloader(mode="val", batch_size=4)

        optimizer = Adam(faster_rcnn_model.parameters(), lr=0.0001)
        device = torch.device("cpu")
        print_freq = 10
        epoch = 0

        metric_logger = train_one_epoch(
            faster_rcnn_model, optimizer, train_dataloader, device, epoch, print_freq
        )

        print(metric_logger)

        torch.save(faster_rcnn_model.state_dict(), "initial_model")
    if sys.argv[1] == "val":
        _, faster_rcnn_model = define_model()
        faster_rcnn_model.load_state_dict(torch.load("initial_model"))

        faster_rcnn_model.eval()
        val_dataloader = setup_dataloader(mode="val", batch_size=4)
        device = torch.device("cpu")
        coco_evaluator = evaluate(faster_rcnn_model, val_dataloader, device)
        print(f"{coco_evaluator=}")
