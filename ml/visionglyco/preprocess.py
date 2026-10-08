"""Image loading and preprocessing shared by training, evaluation and prediction."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .config import CROP_FRACTION, IMAGE_SIZE


def load_eye_image(path, image_size=IMAGE_SIZE, crop_fraction=CROP_FRACTION):
    """Return an RGB uint8 array of shape (image_size, image_size, 3).

    Steps, mirrored in mobile/src/ml/preprocess.ts:
      1. apply EXIF orientation (phone photos are often stored rotated)
      2. centre-crop a square of `crop_fraction` x the shorter side
      3. bilinear resize to image_size x image_size
    Pixel values stay in 0-255; the Keras EfficientNet backbone rescales internally.
    """
    with Image.open(Path(path)) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        width, height = img.size
        side = int(round(min(width, height) * crop_fraction))
        left = (width - side) // 2
        top = (height - side) // 2
        img = img.crop((left, top, left + side, top + side))
        img = img.resize((image_size, image_size), Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)
