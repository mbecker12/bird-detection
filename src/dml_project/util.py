"""
A collection of utility functions
"""
import os
from typing import List
from cv2 import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from glob import glob
import numpy as np
import torch

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

def load_bbox_file(img_path):
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

        # print(f"{boxes=}")
        
    else:
        boxes = []
        cls_indices = []

    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    cls_indices = torch.as_tensor(cls_indices, dtype=torch.float32)
    target["boxes"] = boxes
    target["labels"] = cls_indices

    return target

def plot_img_and_boxes(img_path: str, cv2_img=None, bboxes: List = None):
    if img_path is not None:
        cv2_img = cv2.imread(img_path)
        bboxes = load_bbox_file(img_path)["boxes"]
    else:
        assert cv2_img is not None
        assert bboxes is not None

    if cv2_img.shape[0] <= 4:
        cv2_img = cv2_img.permute(1,2,0)

    cv2_shape = cv2_img.shape

    fig, ax = plt.subplots()

    for box in bboxes:                
        width = cv2_shape[1] * box[2]
        height = cv2_shape[0] * box[3]

        # recompute x and y, since the boxes are defined as in yolo format
        # with x and y being the box center
        # but matplotlib expects x and y to be the lower left corner
        x = cv2_shape[1] * box[0] - 0.5 * width
        y = cv2_shape[0] * box[1] - 0.5 * height

        rect = patches.Rectangle((x,y), width, height, fill=False, color="#FFC0CB")
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

                rect = patches.Rectangle((x,y), width, height, fill=False)
                ax.add_patch(rect)    

            show_img(img)

    # plot_img_and_boxes(path_to_image)
    # plot_img_and_boxes(None, cv2_image_array, list_of_boxes)