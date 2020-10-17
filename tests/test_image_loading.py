"""
Test loading of images.
Make sure that they all have the same datatype
and pixel range.
"""


def test_load_full_image():
    """
    Load a few full images and make sure they are all of the same datatype.
    """
    pass


def test_load_image_and_bbox():
    """
    Load image and bounding box and assure that bbox location corresponds to the correct loaction.
    """
    pass


def test_resize_image():
    """
    Load an image and resize it to the desired size.
    Make sure that the datatype does not change unexpectedly.
    """
    pass


def test_resize_image_and_bbox():
    """
    Load an image, transform it.
    Apply the same transformation to the bounding boxes
    and make sure that the bounding boxes still correspond to the correct objects in the image.
    """
    pass


def test_normalization():
    """
    Make sure that we find a suitable set of normalization parameters across all images, for all channels
    such that the pixel values for the images will be centered and of unity variance.
    """
    pass


def test_resize_and_normalize_with_bbox():
    """
    Load an image, resize it and normalize it with our defined normalization parameters.
    Assure that the transformations behave nominally, as above. Additionally, the pixel values should be
    centered now.
    """
    pass
