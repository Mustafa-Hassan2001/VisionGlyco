"""Generate a SYNTHETIC eye-like dataset for testing the pipeline end to end.

The images are drawn circles whose colour depends on the label, so a model can
learn them easily. They say nothing about real eyes or real glucose: use them
only to check that training, export and the mobile app run. Pass
--synthetic-data to train.py so the app labels the model as synthetic.

    python tools/make_synthetic_dataset.py --out data/synthetic --patients 60
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def draw_eye(rng, positive, size):
    img = Image.new("RGB", (size, size), tuple(int(v) for v in rng.integers(200, 240, 3)))
    draw = ImageDraw.Draw(img)
    cx, cy = (size // 2 + int(rng.integers(-size // 20, size // 20)) for _ in range(2))
    iris_r = int(size * rng.uniform(0.22, 0.28))
    base = np.array([60, 110, 150]) if positive else np.array([120, 80, 40])
    colour = tuple(int(v) for v in np.clip(base + rng.integers(-25, 25, 3), 0, 255))
    draw.ellipse((cx - iris_r, cy - iris_r, cx + iris_r, cy + iris_r), fill=colour)
    pupil_r = int(iris_r * rng.uniform(0.3, 0.45))
    draw.ellipse((cx - pupil_r, cy - pupil_r, cx + pupil_r, cy + pupil_r), fill=(10, 10, 10))
    noise = rng.normal(0, 8, (size, size, 3))
    return Image.fromarray(np.clip(np.asarray(img) + noise, 0, 255).astype(np.uint8))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--patients", type=int, default=60)
    p.add_argument("--images-per-patient", type=int, default=3)
    p.add_argument("--size", type=int, default=320)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for pid in range(args.patients):
        hba1c = float(np.round(rng.uniform(4.8, 9.5), 1))
        for k in range(args.images_per_patient):
            name = f"p{pid:04d}_{k}.jpg"
            draw_eye(rng, hba1c >= 6.5, args.size).save(out / "images" / name, quality=92)
            rows.append({"image": name, "patient_id": f"P{pid:04d}", "hba1c": hba1c})
    pd.DataFrame(rows).to_csv(out / "labels.csv", index=False)
    print(f"Wrote {len(rows)} synthetic images to {out}")


if __name__ == "__main__":
    main()
