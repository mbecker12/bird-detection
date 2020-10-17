"""
Test pytorch data loaders, including data augmentations.
"""


def test_training_loader():
    """
    Test a data loader with the corresponding augmentation pipeline for the training set.
    i.e. the example images and bounding boxes will go through a series of augmentations.
    Make sure that it will return the correct batch size and data format, as well as pixel range.
    """
    pass


def test_validation_loader():
    """
    Test a data loader with the corresponding augmentation pipeline for the validation set.
    These images will not be augmented, only resized and normalized.
    Make sure that it will return the correct batch size and data format, as well as pixel range.
    """
    pass
