"""
Test loading of images.
Make sure that they all have the same datatype
and pixel range.
"""
from cv2 import cv2
from glob import glob
from dml_project.util import show_img, load_images, load_bbox_file, plot_img_and_boxes
from dml_project.const import *
from torchvision import transforms
import albumentations as a
from albumentations.pytorch import ToTensorV2
from dml_project.augmentation import AlbumentationsDatasetCV2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch


def test_load_full_image():
    """
    Load a few full images and make sure they are all of the same datatype.
    """
    images = load_images(DATA_PATH, num_jpg=5, num_png=5)

    loaded_dtype = None
    for i, img in enumerate(images):
        cv2_img = cv2.imread(img)

        if i < 1:
            loaded_dtype = cv2_img.dtype

        new_loaded_dtype = cv2_img.dtype
        assert loaded_dtype == new_loaded_dtype

        cv2_shape = cv2_img.shape
        assert len(cv2_shape) == 3
        assert cv2_shape[0] > MIN_HEIGHT
        assert cv2_shape[1] > MIN_WIDTH
        assert cv2_shape[2] == NUM_CHANNELS


def test_load_image_and_bbox():
    """
    Load image and bounding box and assure that bbox location corresponds to the correct loaction.
    """
    images = load_images(DATA_PATH, num_jpg=5, num_png=5)

    for img in images:
        cv2_img = cv2.imread(img)
        target = load_bbox_file(img)
        bboxes = target["boxes"]
        labels = target["labels"]

        cv2_shape = cv2_img.shape

        for j, box in enumerate(bboxes):
            assert labels[j] < NUM_CLASSES
            assert 0 <= box[0] <= 1
            assert 0 <= box[1] <= 1
            assert 0 <= box[2] <= 1
            assert 0 <= box[3] <= 1

            width = cv2_shape[1] * box[2]
            height = cv2_shape[0] * box[3]

            x = cv2_shape[1] * box[0]
            y = cv2_shape[0] * box[1]

            assert x <= cv2_shape[1]
            assert y <= cv2_shape[0]
            assert x + 0.5 * width <= cv2_shape[1]
            assert y + 0.5 * height <= cv2_shape[1]
            assert x - 0.5 * width >= 0
            assert y - 0.5 * height >= 0


def test_resize_image():
    """
    Load an image and resize it to the desired size.
    Make sure that the datatype does not change unexpectedly.
    """

    image_paths = load_images(DATA_PATH, num_jpg=3, num_png=3)

    albumentations_transform = a.Compose(
        [
            a.Resize(512, 512),
            ToTensorV2(),
        ],
        bbox_params=a.BboxParams(format="yolo", label_fields=["category_ids"]),
    )

    dataset = AlbumentationsDatasetCV2(
        file_paths=image_paths,
        transform=albumentations_transform,
    )

    for i, img in enumerate(image_paths):

        dat_img, boxes, cid = dataset[i]

        _dat_img = dat_img.permute(1, 2, 0)

        cv2_img = cv2.imread(img)
        target = load_bbox_file(img)
        bboxes = target["boxes"]
        labels = target["labels"]

        cv2_shape = cv2_img.shape
        for j, cv2_box in enumerate(bboxes):
            assert labels[j] < NUM_CLASSES
            assert 0 <= cv2_box[0] <= 1
            assert 0 <= cv2_box[1] <= 1
            assert 0 <= cv2_box[2] <= 1
            assert 0 <= cv2_box[3] <= 1

            width = cv2_shape[1] * cv2_box[2]
            height = cv2_shape[0] * cv2_box[3]

            x = cv2_shape[1] * cv2_box[0]
            y = cv2_shape[0] * cv2_box[1]

            assert x <= cv2_shape[1]
            assert y <= cv2_shape[0]
            assert x + 0.5 * width <= cv2_shape[1]
            assert y + 0.5 * height <= cv2_shape[1]
            assert x - 0.5 * width >= 0
            assert y - 0.5 * height >= 0

        dat_shape = _dat_img.shape
        for j, dat_box in enumerate(boxes):
            assert cid[j] < NUM_CLASSES
            assert 0 <= dat_box[0] <= 1
            assert 0 <= dat_box[1] <= 1
            assert 0 <= dat_box[2] <= 1
            assert 0 <= dat_box[3] <= 1

            dat_width = dat_shape[1] * dat_box[2]
            dat_height = dat_shape[0] * dat_box[3]

            dat_x = dat_shape[1] * dat_box[0]
            dat_y = dat_shape[0] * dat_box[1]

            assert dat_x <= dat_shape[1]
            assert dat_y <= dat_shape[0]
            assert dat_x + 0.5 * dat_width <= dat_shape[1]
            assert dat_y + 0.5 * dat_height <= dat_shape[1]
            assert dat_x - 0.5 * dat_width >= 0
            assert dat_y - 0.5 * dat_height >= 0


def test_normalization():
    """
    Make sure that we find a suitable set of normalization parameters across all images, for all channels
    such that the pixel values for the images will be centered and of unity variance.
    """

    image_paths = load_images(DATA_PATH, num_jpg=3, num_png=3)

    albumentations_transform = a.Compose(
        [
            a.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
            ToTensorV2(),
        ],
        bbox_params=a.BboxParams(format="yolo", label_fields=["category_ids"]),
    )

    dataset = AlbumentationsDatasetCV2(
        file_paths=image_paths,
        transform=albumentations_transform,
    )

    for img, boxes, labels in dataset:
        assert img.dtype in (np.float32, float, torch.float, torch.float32)
        assert -3 < img.min() < 0
        assert 0 < img.max() < 3


def test_resize_and_normalize_with_bbox():
    """
    Load an image, resize it and normalize it with our defined normalization parameters.
    Assure that the transformations behave nominally, as above. Additionally, the pixel values should be
    centered now.
    """
    image_paths = load_images(DATA_PATH, num_jpg=3, num_png=3)

    resize_width = 512
    resize_height = 512
    albumentations_transform = a.Compose(
        [
            a.Resize(resize_height, resize_width),
            a.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
            ToTensorV2(),
        ],
        bbox_params=a.BboxParams(format="yolo", label_fields=["category_ids"]),
    )

    dataset = AlbumentationsDatasetCV2(
        file_paths=image_paths,
        transform=albumentations_transform,
    )

    assert len(dataset) <= 6

    for img, boxes, labels in dataset:
        assert img.dtype in (np.float32, float, torch.float, torch.float32)
        assert -3 < img.min() < 0
        assert 0 < img.max() < 3

        img_shape = img.shape
        assert img_shape[0] == 3
        assert img_shape[1] == resize_height
        assert img_shape[2] == resize_width


def test_dataset_wo_transform():
    image_paths = load_images(DATA_PATH, num_jpg=3, num_png=3)

    dataset = AlbumentationsDatasetCV2(file_paths=image_paths)

    for img, boxes, labels in dataset:
        assert boxes is not None
        assert labels is not None
        assert len(img.shape) == 3


def test_load_all_images():
    image_paths = load_images("data")

    dataset = AlbumentationsDatasetCV2(file_paths=image_paths)

    for i, (img, boxes, labels) in enumerate(dataset):
        if i > 10:
            break

        assert boxes is not None
        assert labels is not None
        assert len(img.shape) == 3

    # load example image without txt file
    image_paths = glob("data/universeum*")

    dataset = AlbumentationsDatasetCV2(file_paths=image_paths)

    for i, (img, boxes, labels) in enumerate(dataset):
        if i > 10:
            break

        assert boxes is not None
        assert labels is not None
        assert len(img.shape) == 3
