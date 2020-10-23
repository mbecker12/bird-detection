from typing import Dict, Union, List
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
import numpy as np
from copy import deepcopy
from dml_project.training_utils import (
    validation_string,
    update_plot_data,
    update_plots,
    setup_plots,
    LOSS_KEYS,
)

sys.path.append(os.getcwd() + "/src/vision")
sys.path.append(os.getcwd() + "/src/vision/references/detection")
from vision.references.detection.engine import train_one_epoch, evaluate
from vision.references.detection.utils import reduce_dict, MetricLogger, SmoothedValue

MODEL_NAME = "NO_NORMALIZE"

def validate(model, data_loader, device = torch.device("cpu")):
    with torch.no_grad():
        model.to(device)
        print_freq = 1

        loss_dicts = [None] * len(data_loader)
        loss_summary = {}
        # for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        for i, (images, targets) in enumerate(data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss_dicts[i] = loss_dict

            for k, v in loss_dict.items():
                if loss_summary.get(k, None) is None:
                    loss_summary[k] = [v.item()]
                else:
                    loss_summary[k].append(v.item())

        loss_summary["loss"] = [None] * len(data_loader)
        for i in range(len(data_loader)):
            loss_summary["loss"][i] = 0
            for k, v in loss_summary.items():
                if k == "loss":
                    continue
                loss_summary["loss"][i] += v[i]

    return loss_dicts, loss_summary


if __name__ == "__main__":
    if sys.argv[1] == "train":
        _, faster_rcnn_model = define_model()

        # ~~~~~~~~~~~~ load image paths ~~~~~~~~~~~ #
        train_paths = load_images(TRAIN_PATH)
        val_paths   = load_images(VAL_PATH)
        test_paths  = load_images(TEST_PATH)

        # ~~~~~~~~~~~~~ load datasets ~~~~~~~~~~~~~ #
        train_dataset = AlbumentationsDatasetCV2(
            file_paths=train_paths,
            transform=albumentations_transform("train", normalize=False),
        )

        val_dataset = AlbumentationsDatasetCV2(
            file_paths=val_paths,
            transform=albumentations_transform("val", normalize=False),
        )

        test_dataset = AlbumentationsDatasetCV2(
            file_paths=test_paths,
            transform=albumentations_transform("val", normalize=False),
        )

        train_dataloader = setup_dataloader(
            dataset=train_dataset, batch_size=2, num_workers=0
        )
        val_dataloader = setup_dataloader(
            dataset=val_dataset, batch_size=2, num_workers=0
        )
        test_dataloader = setup_dataloader(
            dataset=test_dataset, batch_size=2, num_workers=0
        )

        params = [p for p in faster_rcnn_model.parameters() if p.requires_grad]
        optimizer = Adam(params, lr=0.0001, weight_decay=0.0005)

        # and a learning rate scheduler which decreases the learning rate by
        # 10x every 3 epochs
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=0.1
        )
        lr_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=10, min_lr=1e-8
        )
        lr_lambda = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda ep: 0.95 ** ep)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print_freq = 10
        num_epochs = 4

        faster_rcnn_model.to(device)

        # setup saving at best validation score
        best_model = None
        best_val_loss = np.inf

        # metric_loggers = [None] * num_epochs
        # val_metrics = [None] * num_epochs
        # val_loss_summaries = [None] * num_epochs

        plot_train_losses = {loss_key: [] for loss_key in LOSS_KEYS}
        plot_valid_losses = {loss_key: [] for loss_key in LOSS_KEYS}
        fig, axs_dict = setup_plots()

        for epoch in range(num_epochs):
            
            metric_logger = train_one_epoch(
                faster_rcnn_model,
                optimizer,
                train_dataloader,
                device,
                epoch,
                print_freq,
            )
            # metric_logger = None  # stand-in for testing val function

            # update the learning rate
            lr_lambda.step()
            # evaluate
            _, val_loss_summary = validate(faster_rcnn_model, val_dataloader, device)
            # TODO: is it better to use median or avg loss
            # this is done for early saving/stopping of training process
            loss = np.median(val_loss_summary["loss"])

            val_string = validation_string(val_loss_summary)
            print(val_string)

            optimizer.zero_grad()
            if loss < best_val_loss:
                best_val_loss = loss
                print(f"{best_val_loss=}")
                best_model = deepcopy(faster_rcnn_model.to(torch.device("cpu")))
                faster_rcnn_model.to(device)

            # val_metrics[epoch] = val_losses
            # metric_loggers[epoch] = metric_logger
            # val_loss_summaries[epoch] = val_loss_summary

            new_train_data = {
                loss_key: metric_logger.meters[loss_key].median
                for loss_key in LOSS_KEYS
            }
            new_valid_data = {
                loss_key: np.median(val_loss_summary[loss_key])
                for loss_key in LOSS_KEYS
            }

            plot_train_losses, plot_valid_losses = update_plot_data(
                plot_train_losses, new_train_data, plot_valid_losses, new_valid_data
            )

            update_plots(
                fig, axs_dict, train_data=plot_train_losses, val_data=plot_valid_losses
            )

        torch.save(best_model.state_dict(), MODEL_NAME)
        plt.savefig(MODEL_NAME + "_training_losses.jpg")

    if sys.argv[1] == "val":
        _, faster_rcnn_model = define_model()
        faster_rcnn_model.load_state_dict(torch.load(MODEL_NAME))

        faster_rcnn_model.eval()
        val_dataloader = setup_dataloader(mode="val", batch_size=4)
        device = torch.device("cpu")
        coco_evaluator = evaluate(faster_rcnn_model, val_dataloader, device)
        print(f"{coco_evaluator=}")

    if sys.argv[1] == "eval":
        _, faster_rcnn_model = define_model()
        faster_rcnn_model.load_state_dict(torch.load(MODEL_NAME))

        val_paths   = load_images(sys.argv[2] if len(sys.argv) > 2 else VAL_PATH)
        val_dataset = AlbumentationsDatasetCV2(
            file_paths=val_paths,
            transform=albumentations_transform("val", normalize=False),
        )
        val_dataloader   = setup_dataloader(dataset=val_dataset,   batch_size=20, num_workers=0)

        faster_rcnn_model.eval()
        with torch.no_grad():
            images, targets = next(iter(val_dataloader))
            outputs = faster_rcnn_model(images)

            for j, outp in enumerate(outputs):
                print(f"{outp=}")
                plot_img_and_boxes(
                    None, images[j], normalize_boxes(outp["boxes"], images[j].shape), [e.item() for e in outp["labels"]]
                )
                plt.show()
                
    if sys.argv[1] == "coco":
        _, faster_rcnn_model = define_model()
        faster_rcnn_model.load_state_dict(torch.load("NO_NORMALIZE"))

        val_paths = load_images(TEST_PATH)
        
        val_dataset = AlbumentationsDatasetCV2(
            file_paths=val_paths,
            transform=albumentations_transform("val", normalize=False),
        )
        val_dataloader = setup_dataloader(
            dataset=val_dataset, batch_size=2, num_workers=0
        )

        device = torch.device("cpu")
        coco_evaluator = evaluate(faster_rcnn_model, val_dataloader, device)
