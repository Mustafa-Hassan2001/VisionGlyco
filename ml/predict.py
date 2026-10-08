"""Run an exported VisionGlyco TFLite model on one or more eye photos.

    python predict.py --model-dir outputs photo1.jpg photo2.jpg
"""

import argparse
import json
from pathlib import Path

import numpy as np

from visionglyco.export import METADATA_FILENAME, MODEL_FILENAME, tflite_predict
from visionglyco.preprocess import load_eye_image


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", default="outputs")
    p.add_argument("images", nargs="+")
    args = p.parse_args(argv)

    model_dir = Path(args.model_dir)
    meta = json.loads((model_dir / METADATA_FILENAME).read_text())
    size, crop = meta["input"]["size"], meta["input"]["crop_fraction"]
    batch = np.stack([load_eye_image(path, size, crop) for path in args.images]).astype(np.float32)
    outputs = tflite_predict(model_dir / MODEL_FILENAME, batch)

    spec = meta["output"]
    results = []
    for path, value in zip(args.images, outputs):
        if spec["type"] == "probability":
            positive = value >= spec["threshold"]
            label = spec["positive_label"] if positive else spec["negative_label"]
            results.append({"image": path, "probability": round(value, 4), "result": label})
        else:
            results.append({"image": path, "value": round(value, 1), "unit": spec["unit"]})
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
