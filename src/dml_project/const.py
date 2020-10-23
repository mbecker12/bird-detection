DATA_PATH = "data/"
TRAIN_PATH = "data/train/"
VAL_PATH   = "data/val/"
TEST_PATH  = "data/test/"

MIN_WIDTH = 800
MIN_HEIGHT = 800
NUM_CHANNELS = 3

NUM_CLASSES = 6

CLASS_NAMES = [
    "Crimson-rumped toucanet",
    "Scarlet Ibis",
    "Other toucan",
    "Sunbittern",
    "Toco toucan",
    "Inca jay"
]

# TODO: models expect pixel values in [0, 1]
# edit: before it's normalizing the images itself
# and not as previously assumed in [-1, 1]

# Also, boxes might need to be redefined
# to 0 < x < W, 0 < y < H


# FasterRCNN takes input arguments
#   image_mean and image_std
# Then, it seems like the images don't need to be
# normalized in the augmentation pipeline,
# but they should be in the range of [0, 1]
#
# for mobilenet_v2:
# mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
