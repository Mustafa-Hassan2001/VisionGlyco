"""Iris unwrapping (polar transform), mirrored in mobile/src/ml/polar.ts.

The input is a square RGB(A) image whose inscribed circle is the iris (the user lines
the iris up with the app's guide circle, or crops a gallery photo to it). The output
matches the strips in the public Farabi Hospital dataset: column j is the ray at
angle j * 360 / width degrees, starting at the right and turning towards the bottom
of the image; row 0 is the pupil edge and the last row is the outer iris edge.
"""

import numpy as np

RING_BINS = 50
IRIS_EDGE = 0.98  # stay just inside the guide circle
PUPIL_MIN, PUPIL_MAX, PUPIL_DEFAULT = 0.15, 0.7, 0.35
MIN_CONTRAST = 10.0
PUPIL_EDGE_FACTOR = 0.1


def _luminance(img):
    img = img.astype(np.float64)
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def estimate_pupil_radius(img):
    """Pupil radius as a fraction of the iris radius.

    Averages brightness in concentric rings around the centre; the pupil edge is the
    first ring, moving out from the darkest one, brighter than the pupil by
    PUPIL_EDGE_FACTOR of the pupil-to-outer-iris difference. (The iris next to the
    pupil is often dark too, so a halfway threshold overshoots.)
    """
    size = img.shape[0]
    centre = (size - 1) / 2
    radius = size / 2
    lum = _luminance(img)
    ys, xs = np.mgrid[0:size, 0:size]
    rel = np.hypot(xs - centre, ys - centre) / radius
    bins = np.floor(rel * RING_BINS).astype(int)
    sums = np.zeros(RING_BINS)
    counts = np.zeros(RING_BINS)
    inside = bins < RING_BINS
    np.add.at(sums, bins[inside], lum[inside])
    np.add.at(counts, bins[inside], 1)
    means = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)

    inner = [k for k in range(RING_BINS) if (k + 0.5) / RING_BINS < 0.6 and counts[k] > 0]
    outer = [k for k in range(RING_BINS) if 0.6 <= (k + 0.5) / RING_BINS < 0.9 and counts[k] > 0]
    darkest = min(inner, key=lambda k: means[k])
    iris_level = float(np.median(means[outer]))
    if iris_level - means[darkest] < MIN_CONTRAST:
        return PUPIL_DEFAULT
    threshold = means[darkest] + PUPIL_EDGE_FACTOR * (iris_level - means[darkest])
    for k in range(darkest, RING_BINS):
        if counts[k] > 0 and means[k] > threshold:
            return float(np.clip(k / RING_BINS, PUPIL_MIN, PUPIL_MAX))
    return PUPIL_DEFAULT


def unwrap_iris(img, pupil_fraction, height, width):
    """Nearest-neighbour polar unwrap; returns float32 (height, width, 3) RGB in 0-255.

    Nearest-neighbour (as in the dataset's own preprocessing) rather than bilinear: the
    model is sensitive to fine pixel texture, and smoothing changes its predictions.
    """
    size = img.shape[0]
    centre = (size - 1) / 2
    radius = size / 2
    theta = np.deg2rad(np.arange(width) * 360.0 / width)
    r = pupil_fraction + (np.arange(height) / (height - 1))[:, None] * (IRIS_EDGE - pupil_fraction)
    # floor(v + 0.5) rather than np.round, to match JavaScript's Math.round exactly.
    x = np.clip(np.floor(centre + r * radius * np.cos(theta)[None, :] + 0.5), 0, size - 1).astype(int)
    y = np.clip(np.floor(centre + r * radius * np.sin(theta)[None, :] + 0.5), 0, size - 1).astype(int)
    return img[y, x, :3].astype(np.float32)


def wrap_strip(strip, size, pupil_fraction, background=(220, 220, 220), pupil_colour=(15, 15, 15)):
    """Inverse of unwrap_iris: draw a strip back into a round eye image (for tests)."""
    height, width = strip.shape[:2]
    centre = (size - 1) / 2
    radius = size / 2
    ys, xs = np.mgrid[0:size, 0:size]
    rel = np.hypot(xs - centre, ys - centre) / radius
    ang = np.mod(np.degrees(np.arctan2(ys - centre, xs - centre)), 360)
    out = np.empty((size, size, 3), dtype=np.float32)
    out[:] = background
    out[rel < pupil_fraction] = pupil_colour
    ring = (rel >= pupil_fraction) & (rel <= IRIS_EDGE)
    rows = np.clip(np.round((rel[ring] - pupil_fraction) / (IRIS_EDGE - pupil_fraction) * (height - 1)), 0, height - 1)
    cols = np.mod(np.round(ang[ring] / 360 * width), width)
    out[ring] = strip[rows.astype(int), cols.astype(int), :3]
    return np.clip(out, 0, 255).astype(np.uint8)
