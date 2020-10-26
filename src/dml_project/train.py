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
from torchvision.ops import nms

sys.path.append(os.getcwd() + "/src/vision")
sys.path.append(os.getcwd() + "/src/vision/references/detection")
# Import PyTorch module "vision" (https://github.com/pytorch/vision)
# which implements the main logic for training
from vision.references.detection.engine import train_one_epoch, evaluate
from vision.references.detection.utils import reduce_dict, MetricLogger, SmoothedValue

MODEL_NAME = "NO_NORMALIZE"

def validate(model, data_loader, device = torch.device("cpu")):
    """
    Compute the validation loss.
    Strongly based on both funtcions train_one_epoch() and evaluate() from 
    vision.references.detection.engine.

    """
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
    """
    Select different options:
    train - perform the training loop. This is where our training process was performed
    eval - load some validation/test images and show predicted bounding boxes
    coco - calculates mAP score, based on https://towardsdatascience.com/evaluating-performance-of-an-object-detection-model-137a349c517b
        this was put in a custom submodule. Since we didn't write this piece of code ourselves, we decided not to include it for the hand-in.
    """
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

        # ~ setup saving at best validation score ~ #
        best_model = None
        best_val_loss = np.inf

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

            # update the learning rate
            lr_lambda.step()
            # evaluate
            _, val_loss_summary = validate(faster_rcnn_model, val_dataloader, device)
            loss = np.median(val_loss_summary["loss"])

            val_string = validation_string(val_loss_summary)
            print(val_string)

            optimizer.zero_grad()
            if loss < best_val_loss:
                best_val_loss = loss
                print(f"{best_val_loss=}")
                best_model = deepcopy(faster_rcnn_model.to(torch.device("cpu")))
                faster_rcnn_model.to(device)

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

    if sys.argv[1] == "eval":
        """
        Feeds the selected model (MODEL_NAME) with images from an image set
        and displays the output bboxes and classifications on the images
        """

        # ~~~~~~~~~~~~~~~ load model ~~~~~~~~~~~~~~ #
        device = torch.device("cuda")
        _, faster_rcnn_model = define_model()
        faster_rcnn_model.load_state_dict(torch.load(MODEL_NAME))
        faster_rcnn_model.to(device)

        # ~~~~~~~~~~~~ init data loader ~~~~~~~~~~~ #
        val_paths   = load_images(sys.argv[2] if len(sys.argv) > 2 else VAL_PATH)
        val_dataset = AlbumentationsDatasetCV2(
            file_paths=val_paths,
            transform=albumentations_transform("val", normalize=False),
        )
        val_dataloader   = setup_dataloader(dataset=val_dataset,   batch_size=10, num_workers=0)

        #  run the network on each image and display the result  #
        faster_rcnn_model.eval()
        with torch.no_grad():
            images, targets = next(iter(val_dataloader))
            outputs = faster_rcnn_model([image.to(device) for image in images])

            for j, outp in enumerate(outputs):                
                keep_indexs = nms(outp["boxes"], outp["scores"], 0.3)

                print(f"{outp=}")
                plot_img_and_boxes(
                    None,
                    images[j].to(torch.device("cpu")),
                    normalize_boxes(outp["boxes"][keep_indexs],
                    images[j].shape),
                    [e.item() for e in outp["labels"][keep_indexs]]
                )
                plt.show()
                
    if sys.argv[1] == "coco":
        """
        Use the code provided by https://towardsdatascience.com/evaluating-performance-of-an-object-detection-model-137a349c517b
        and calculate the mAP score based on the recall-precision curve explained in the same article.
        The module 'evaluation' which is imported at the top of this section, contains the code from this article.
        """

        from evaluation.eval import map_evaluation
        _, faster_rcnn_model = define_model()
        faster_rcnn_model.load_state_dict(torch.load(MODEL_NAME))

        val_paths = load_images(sys.argv[2] if len(sys.argv) > 2 else TEST_PATH)
        
        val_dataset = AlbumentationsDatasetCV2(
            file_paths=val_paths,
            transform=albumentations_transform("val", normalize=False),
        )
        val_dataloader = setup_dataloader(
            dataset=val_dataset, batch_size=2, num_workers=0
        )

        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        results = map_evaluation(faster_rcnn_model, val_dataloader, device, iou_thr=0.5)

        import json
        import os
        os.makedirs(f"results", exist_ok=True)
        os.makedirs(f"results/{model_name}", exist_ok=True)
        for iou in results.keys():
            print(f"Average Precision @ {iou}: {results[iou]['avg_prec']}")
            plt.scatter(results[iou]["recalls"][::-1], results[iou]["precisions"][::-1])
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(model_name + f" iou {iou}")
            plt.show()

            for k in results[iou].keys():
                with open(f"results/{model_name}/{iou}_{k}", "w") as results_file:
                    if "avg" in k:
                        results_file.write(str(results[iou][k]))
                    else:
                        json.dump(list(results[iou][k]), results_file)