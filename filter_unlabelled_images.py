"""
Read through the list of images.
For each image, there should exist a corresponding txt file with the same name as the image
sans the file ending.

This program looks out for images without such a txt file and asks the user if
they want to add annotations through 'labelImg', using the available 'classes.txt' file
in DATA_PATH.

Otherwise, if it turns out that the image is useless and therefore it was decided to not label it,
the user can dcide to remove the image altogether.

"""

import subprocess
import os
from glob import glob

DATA_PATH = "./data/"

for i, filepath in enumerate(glob(DATA_PATH + "*")):
    full_filename = filepath.split("/")[-1]
    # print(full_filename)

    filename, ending = full_filename.split(".")
    if ending == "txt":
        continue

    if ending == "jpg":
        # print(f"Corresponding txt file: {DATA_PATH + filename + '.txt'}")
        if not os.path.exists(DATA_PATH + filename + ".txt"):
            print(f"This file would be removed: {DATA_PATH + full_filename}")

            want_edit = input("Do you want to add annotations for this file?")

            if want_edit == "y" or want_edit == "Y":
                subprocess.Popen(
                    f"labelImg {DATA_PATH + full_filename} {DATA_PATH}classes.txt",
                    shell=True,
                )

            want_delete = input("Do you want to delete the image?")

            if want_delete == "y" or want_delete == "Y":
                assured = input("Are you sure?")
                if assured == "y" or assured == "Y":
                    os.remove(DATA_PATH + full_filename)
