"""
Implement a model for object detection, transfer learning for small sample sizes.
Based on this example:
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html?fbclid=IwAR2sfSCFONa6C9iwL6BHtS0Ruc4B_BDKuntJYGrfPv4NZFBkZCXgPN8-Fv8
"""

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from dml_project.augmentation import AlbumentationsDatasetCV2, albumentations_transform
from dml_project.const import NUM_CHANNELS, NUM_CLASSES
from glob import glob
from dml_project.const import *
from dml_project.util import *
from torch.utils.data import DataLoader
import torch


def define_model(resnet=False):
    model = None
    # load a model pre-trained pre-trained on COCO
    if resnet:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    if resnet:
        in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    if resnet:
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    # load a pre-trained model for classification and return
    # only the features
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features

    # FasterRCNN needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280
    # so we need to add it here
    backbone.out_channels = 1280

    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios

    # deleted another entry with size 512, taken from the example
    # didn't seem to match with our used image sizes
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256),), aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0"],
        output_size=7,
        sampling_ratio=2,
    )

    # put the pieces together inside a FasterRCNN model
    faster_rcnn_model = FasterRCNN(
        backbone,
        num_classes=NUM_CLASSES + 1,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
    )

    return model, faster_rcnn_model


def custom_collate_fn(loaded_data):
    batch_size = len(loaded_data)

    imgs = [
        torch.as_tensor(loaded_data[i][0], dtype=torch.float32)
        for i in range(batch_size)
    ]

    targets = []
    for i in range(batch_size):
        boxes = recompute_boxes(loaded_data[i][1]["boxes"], imgs[i].shape)
        targets.append(
            {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                "labels": torch.as_tensor(loaded_data[i][1]["labels"]).to(
                    dtype=torch.int64
                ),
            }
        )

    img_batch_tensor = torch.stack(imgs, dim=0).to(dtype=torch.float32)

    assert (
        img_batch_tensor.shape[0] == batch_size
    ), f"{img_batch_tensor.shape[0]=}, {batch_size=}"
    assert (
        img_batch_tensor.shape[1] == NUM_CHANNELS
    ), f"{img_batch_tensor.shape[1]=}, {NUM_CHANNELS=}"
    return (img_batch_tensor, targets)


def custom_collate_fn_mem_efficient(loaded_data):
    batch_size = len(loaded_data)

    boxes = [
        recompute_boxes(loaded_data[i][1]["boxes"], loaded_data[i][0].shape)
        for i in range(batch_size)
    ]

    batch_areas = []
    for i in range(batch_size):
        areas = torch.ones_like(boxes[i])
        for j, a in enumerate(areas):
            areas[j] *= (boxes[i][j][1] - boxes[i][j][0]) * (
                boxes[i][j][3] - boxes[i][j][2]
            )
        batch_areas.append(areas)

    return (
        torch.stack(
            [
                torch.as_tensor(loaded_data[i][0], dtype=torch.float32)
                for i in range(batch_size)
            ],
            dim=0,
        ).to(dtype=torch.float32),
        [
            {
                "boxes": torch.as_tensor(boxes[i], dtype=torch.float32).reshape(-1, 4),
                "labels": torch.as_tensor(loaded_data[i][1]["labels"]).to(
                    dtype=torch.int64
                ),
                "image_id": torch.as_tensor(loaded_data[i][1]["image_id"]),
                "iscrowd": torch.as_tensor(loaded_data[i][1]["iscrowd"]),
                "area": torch.as_tensor(batch_areas[i]),  # this works
            }
            for i in range(batch_size)
        ],
    )


def setup_dataloader(
    mode, batch_size=16, num_workers=0, shuffle=True, num_jpg=-1, num_png=-1
):
    dataset = AlbumentationsDatasetCV2(
        file_paths=load_images(DATA_PATH, num_jpg=num_jpg, num_png=num_png),
        transform=albumentations_transform(mode),
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=custom_collate_fn_mem_efficient,
    )

    return data_loader
