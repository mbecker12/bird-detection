"""
Test the application of a full data augmentation pipeline.
Test the loading, resizing, transformation, augmentation of images;
and possibly changing its datatype to one that is needed by pytorch.
"""


def test_augmentation_pipeline_only_img():
    """
    Check that a predefined augmentation pipeline behaves as expected on few example images (without bboxes).
    Assure that datatypes stay as expected and is in a pytorch-usable format at the end.
    """
    pass


def test_augmentation_pipeline_with_bbox():
    """
    Check that a predefined augmentation pipeline behaves as expected on few example images and their bounding boxes.
    Assure that datatypes stay as expected and is in a pytorch-usable format at the end.
    Also assure that the bounding boxes still correspond to their intended objects.
    """
    pass
