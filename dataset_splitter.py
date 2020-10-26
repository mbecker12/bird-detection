from os import walk, sep, mkdir
from os.path import exists
import sys
from shutil import move
import numpy as np
from glob import glob

def split_into_train_val_test(path, ratios = (0.75, 0.15, 0.1)):
    if path[-1] != sep:
        path = path + sep

    # ~~~~~~~~~~~~~ load all paths ~~~~~~~~~~~~ #
    filenames = glob(path + "*.*") # To ignore folders
    unique_names = list(set(filename.split(".")[0] for filename in filenames))

    # ~~~ split names into train, val, test ~~~ #
    np.random.shuffle(unique_names)
    train_ratio, val_ratio, _ = ratios
    train_count = int(train_ratio * len(unique_names))
    val_count = int(val_ratio * len(unique_names))

    train_names = unique_names[:train_count]
    val_names = unique_names[train_count : train_count + val_count]
    test_names = unique_names[train_count + val_count:]

    # ~~~~~~~~~~~~~ create folders ~~~~~~~~~~~~ #
    if not exists(path + "train"):
        mkdir(path + "train")
    if not exists(path + "val"):
        mkdir(path + "val")
    if not exists(path + "test"):
        mkdir(path + "test")

    # ~~~ Move files to respective directory ~~ #
    for data_names, folder_name in [(train_names, "train"), (val_names, "val"), (test_names, "test")]:
        for file_name in data_names:
            dest_folder = path + folder_name + sep
            for file_path in glob(file_name + "*"):
                dst = dest_folder + file_path.split(sep)[-1]
                move(file_path, dst)
        

if __name__ == "__main__":
    split_into_train_val_test(sys.argv[1])