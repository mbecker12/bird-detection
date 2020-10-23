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

# import plotly.graph_objects as go


sys.path.append(os.getcwd() + "/src/vision")
sys.path.append(os.getcwd() + "/src/vision/references/detection")
from vision.references.detection.engine import train_one_epoch, evaluate
from vision.references.detection.utils import reduce_dict, MetricLogger, SmoothedValue

def validate(model, data_loader):
    with torch.no_grad():
        model.eval()
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)

        device = torch.device("cpu")
        model.to(device)
        print_freq = 1

        val_losses = [None] * len(data_loader)
        for i, (images, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)[0]

            print(f"{loss_dict=}")
            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            val_losses[i] = losses_reduced
    
    return val_losses
    


if __name__ == "__main__":
    if sys.argv[1] == "train":
        _, faster_rcnn_model = define_model()
        
        # ~~~~~~~~~~~~ load image paths ~~~~~~~~~~~ #
        all_img_paths = load_images(DATA_PATH)

        # ~~~ split paths into train, val, test ~~~ #
        np.random.shuffle(all_img_paths)
        train_ratio, val_ratio, test_ratio = 0.75, 0.15, 0.1
        train_count = int(train_ratio * len(all_img_paths))
        val_count = int(val_ratio * len(all_img_paths))

        train_paths = all_img_paths[:train_count]
        val_paths = all_img_paths[train_count : train_count + val_count]
        test_paths = all_img_paths[train_count + val_count:]
        
        # ~~~~~~~~~~~~~ load datasets ~~~~~~~~~~~~~ #
        train_dataset = AlbumentationsDatasetCV2(
            file_paths=train_paths,
            transform=albumentations_transform("train"),
        )

        val_dataset = AlbumentationsDatasetCV2(
            file_paths=val_paths,
            transform=albumentations_transform("val"),
        )

        test_dataset = AlbumentationsDatasetCV2(
            file_paths=test_paths,
            transform=albumentations_transform("val"),
        )

        train_dataloader = setup_dataloader(dataset=train_dataset, batch_size=2, num_workers=0)
        val_dataloader   = setup_dataloader(dataset=val_dataset,   batch_size=2, num_workers=0)
        test_dataloader  = setup_dataloader(dataset=test_dataset,  batch_size=2, num_workers=0)

        params = [p for p in faster_rcnn_model.parameters() if p.requires_grad]
        optimizer = Adam(params, lr=0.0001, weight_decay=0.0005)
        
        # and a learning rate scheduler which decreases the learning rate by
        # 10x every 3 epochs
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        lr_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            factor=0.1, 
            patience=10, 
            min_lr=1e-8
        )
        lr_lambda = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda ep: 0.95 ** ep)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print_freq = 10
        num_epochs = 4

        faster_rcnn_model.to(device)
        
        # setup saving at best validation score
        # ...
        metric_loggers = [None] * num_epochs
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
            # lr_scheduler.step()
            # evaluate
            # val_losses = validate(faster_rcnn_model, val_dataloader)
            # print(f"{val_losses=}")
            metric_loggers[epoch] = metric_logger

        print(f"{metric_logger.meters.items()=}")

        torch.save(faster_rcnn_model.state_dict(), "second_model")
        
    if sys.argv[1] == "val":
        _, faster_rcnn_model = define_model()
        faster_rcnn_model.load_state_dict(torch.load("second_model"))

        faster_rcnn_model.eval()
        val_dataloader = setup_dataloader(mode="val", batch_size=4)
        device = torch.device("cpu")
        coco_evaluator = evaluate(faster_rcnn_model, val_dataloader, device)
        print(f"{coco_evaluator=}")

    if sys.argv[1] == "eval":
        _, faster_rcnn_model = define_model()
        faster_rcnn_model.load_state_dict(torch.load("second_model"))
        val_dataloader = setup_dataloader(mode="val", batch_size=6)

        faster_rcnn_model.eval()
        with torch.no_grad():
            images, targets = next(iter(val_dataloader))
            outputs = faster_rcnn_model(images)

            for j, outp in enumerate(outputs):
                plot_img_and_boxes(
                    None, images[j], normalize_boxes(outp["boxes"], images[j].shape)
                )
                plt.show()