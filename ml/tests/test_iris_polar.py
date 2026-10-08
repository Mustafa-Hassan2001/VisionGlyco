import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visionglyco.iris_polar import estimate_pupil_radius, unwrap_iris, wrap_strip  # noqa: E402


def test_round_trip_recovers_strip_and_pupil():
    rng = np.random.default_rng(0)
    strip = rng.integers(60, 200, (21, 72, 3)).astype(np.uint8)
    eye = wrap_strip(strip, 300, 0.3)
    pupil = estimate_pupil_radius(eye)
    assert abs(pupil - 0.3) <= 0.03
    back = unwrap_iris(eye, 0.3, 21, 72)
    # Nearest-neighbour both ways: most pixels come back exactly.
    assert np.mean(np.all(back[1:] == strip[1:], axis=2)) > 0.8


def test_featureless_image_uses_default_pupil():
    assert estimate_pupil_radius(np.full((64, 64, 3), 128, np.uint8)) == 0.35


def test_unwrap_direction():
    size = 100
    c = (size - 1) / 2
    ys, xs = np.mgrid[0:size, 0:size]
    img = np.where((xs > c)[..., None], [0, 0, 255], [255, 0, 0]).astype(np.uint8)
    strip = unwrap_iris(img, 0.2, 11, 36)
    assert strip.shape == (11, 36, 3)
    assert list(strip[5, 0]) == [0, 0, 255]   # 0 degrees: right
    assert list(strip[5, 18]) == [255, 0, 0]  # 180 degrees: left
