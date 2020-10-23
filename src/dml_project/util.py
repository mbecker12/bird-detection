"""
A collection of utility functions
"""
import os
from typing import List, Union, Tuple
from cv2 import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from glob import glob
import numpy as np
import torch
from dml_project.const import CLASS_NAMES


def show_img(img_path: str):
    cv2_img = cv2.imread(img_path)
    normal_image = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    plt.imshow(normal_image)


def load_images(path: str, num_jpg: int = -1, num_png: int = -1) -> List:
    if path[-1] != "/":
        path += "/"

    if num_jpg > -1:
        jpg_list = glob(path + "*.jpg")[:num_jpg]
    else:
        jpg_list = glob(path + "*.jpg")

    if num_png > -1:
        png_list = glob(path + "*.png")[:num_png]
    else:
        png_list = glob(path + "*.png")

    images = jpg_list + png_list
    return images


def normalize_boxes(boxes: List[Tuple], img_shape: Union[Tuple, List]) -> List[Tuple]:
    img_height = img_shape[1]
    img_width = img_shape[2]

    boxes_ = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        width = x2 - x1
        height = y2 - y1
        x_mid = x1 + 0.5 * width
        y_mid = y1 + 0.5 * height

        box = [
            x_mid / img_width,
            y_mid / img_height,
            width / img_width,
            height / img_height,
        ]
        boxes_.append(box)
    return boxes_


def recompute_boxes(boxes: List[Tuple], img_shape: Union[Tuple, List]) -> List[Tuple]:
    if img_shape is None:
        img_width = 1
        img_height = 1
    else:
        img_height = img_shape[1]
        img_width = img_shape[2]

    for i in range(len(boxes)):
        x, y, width, height = boxes[i]
        x1 = max((x - 0.5 * width) * img_width, 0)
        y1 = max((y - 0.5 * height) * img_height, 0)
        x2 = min((x + 0.5 * width) * img_width, img_width)
        y2 = min((y + 0.5 * height) * img_height, img_height)
        boxes[i] = torch.as_tensor((x1, y1, x2, y2))

    return torch.as_tensor(boxes).reshape(-1, 4)


def load_bbox_file(img_path, img_shape=None):
    bbox_path = img_path.split(".")[-2]
    bbox_path += ".txt"

    target = {}
    boxes = []
    cls_indices = []
    if os.path.isfile(bbox_path):
        with open(bbox_path, "r") as bbox_file:
            for line in bbox_file.readlines():
                cls_idx, x, y, width, height = line.split(" ")
                boxes.append([float(x), float(y), float(width), float(height)])
                cls_indices.append([int(cls_idx)])

    else:
        boxes = []
        cls_indices = []

    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    cls_indices = torch.as_tensor(cls_indices, dtype=torch.float32)
    target["boxes"] = boxes
    target["labels"] = cls_indices
    # NOTE: keep hash as long as img_id is actually uninteresting for us
    target["image_id"] = torch.as_tensor(hash(img_path.split(".")[-2].split("/")[-1]))
    target["iscrowd"] = torch.as_tensor([0 for _ in cls_indices])
    target["area"] = boxes[:, 2] * boxes[:, 3]

    return target


def plot_img_and_boxes(img_path: str, cv2_img=None, bboxes: List = None, label_ids: List = None):
    if img_path is not None:
        cv2_img = cv2.imread(img_path)
        bboxes = load_bbox_file(img_path)["boxes"]
    else:
        assert cv2_img is not None
        assert bboxes is not None

    if cv2_img.shape[0] <= 4:
        cv2_img = cv2_img.permute(1, 2, 0)

    cv2_shape = cv2_img.shape

    fig, ax = plt.subplots()

    for i, box in enumerate(bboxes):
        label_id = label_ids[i]
        width = cv2_shape[1] * box[2]
        height = cv2_shape[0] * box[3]

        # recompute x and y, since the boxes are defined as in yolo format
        # with x and y being the box center
        # but matplotlib expects x and y to be the lower left corner
        x = cv2_shape[1] * box[0] - 0.5 * width
        y = cv2_shape[0] * box[1] - 0.5 * height

        plt.text(x, y, CLASS_NAMES[label_id], color="#FFC0CB")
        rect = patches.Rectangle((x, y), width, height, fill=False, color="#FFC0CB")
        ax.add_patch(rect)

    if img_path is not None:
        normal_image = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

        plt.imshow(normal_image)
    else:
        plt.imshow(cv2_img)

    return fig


if __name__ == "__main__":
    from const import DATA_PATH
    from glob import glob
    import sys

    if sys.argv[1] == "show_images":
        images = load_images(DATA_PATH, num_jpg=5, num_png=5)

        for img in images:
            show_img(img)

    if sys.argv[1] == "test-boxes":
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt

        images = load_images(DATA_PATH, num_jpg=5, num_png=5)

        for img in images:
            cv2_img = cv2.imread(img)
            bboxes = load_bbox_file(img)

            cv2_shape = cv2_img.shape

            fig, ax = plt.subplots()

            for box in bboxes:
                width = cv2_shape[1] * box[3]
                height = cv2_shape[0] * box[4]

                # recompute x and y, since the boxes are defined as in yolo format
                # with x and y being the box center
                # but matplotlib expects x and y to be the lower left corner
                x = cv2_shape[1] * box[1] - 0.5 * width
                y = cv2_shape[0] * box[2] - 0.5 * height

                rect = patches.Rectangle((x, y), width, height, fill=False)
                ax.add_patch(rect)

            show_img(img)

    # plot_img_and_boxes(path_to_image)
    # plot_img_and_boxes(None, cv2_image_array, list_of_boxes)
