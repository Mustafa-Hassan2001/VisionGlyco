"""Train, evaluate and export the VisionGlyco eye-image model.

Example (HbA1c screening, the recommended task):

    python train.py --labels data/labels.csv --images-dir data/images \\
        --task classify --target-column hba1c --threshold 6.5 \\
        --install-to-app ../mobile

Every metric this script reports is measured on patients the model never saw
during training or threshold selection.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import keras
import numpy as np

from visionglyco.config import (
    CROP_FRACTION,
    DEFAULT_TARGET_COLUMN,
    DEFAULT_THRESHOLD,
    IMAGE_SIZE,
    TASK_CLASSIFY,
    TASK_REGRESS,
)
from visionglyco.data import load_labels, make_dataset, split_by_patient
from visionglyco.export import (
    METADATA_FILENAME,
    MODEL_FILENAME,
    check_parity,
    export_tflite,
    install_to_app,
    write_metadata,
)
from visionglyco.metrics import choose_threshold, classification_metrics, regression_metrics
from visionglyco.model import BACKBONES, build_model, compile_model, unfreeze_top

MIN_RECOMMENDED_PATIENTS = 200


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", required=True, help="CSV with columns image, patient_id, <target-column>")
    p.add_argument("--images-dir", required=True, help="Folder the CSV image paths are relative to")
    p.add_argument("--output-dir", default="outputs", help="Where the model and metrics are written")
    p.add_argument("--task", choices=[TASK_CLASSIFY, TASK_REGRESS], default=TASK_CLASSIFY)
    p.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help="classify: target value at or above which a sample is positive")
    p.add_argument("--unit", default="%", help="Unit of the target column, shown in the app")
    p.add_argument("--backbone", choices=sorted(BACKBONES), default="efficientnet_b0")
    p.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    p.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    p.add_argument("--crop-fraction", type=float, default=CROP_FRACTION)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs-head", type=int, default=15)
    p.add_argument("--epochs-finetune", type=int, default=30)
    p.add_argument("--finetune-layers", type=int, default=40)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-finetune", type=float, default=1e-5)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-accuracy", type=float, default=0.85,
                   help="classify: report whether held-out accuracy reached this")
    p.add_argument("--install-to-app", metavar="MOBILE_DIR",
                   help="Copy the exported model into the React Native app")
    p.add_argument("--synthetic-data", action="store_true",
                   help="Mark the model as trained on synthetic data (the app shows a warning)")
    return p.parse_args(argv)


def summarise_splits(splits, task):
    summary = {}
    for name, split in splits.items():
        info = {"images": len(split), "patients": int(len(set(split.patients)))}
        if task == TASK_CLASSIFY:
            info["positive_images"] = int(split.labels.sum())
        summary[name] = info
        print(f"  {name:5s}: {info}")
    return summary


def class_weights(labels):
    counts = np.bincount(labels, minlength=2)
    if counts.min() == 0:
        return None
    total = counts.sum()
    return {i: float(total / (2 * c)) for i, c in enumerate(counts)}


def fit(model, train_ds, val_ds, epochs, task, weights, checkpoint):
    if epochs <= 0:
        return
    monitor, mode = ("val_auc", "max") if task == TASK_CLASSIFY else ("val_mae", "min")
    callbacks = [
        keras.callbacks.EarlyStopping(monitor=monitor, mode=mode, patience=6, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor=monitor, mode=mode, factor=0.3, patience=3),
        keras.callbacks.ModelCheckpoint(str(checkpoint), monitor=monitor, mode=mode, save_best_only=True),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=weights,
              callbacks=callbacks, verbose=2)


def collect_images(ds, limit):
    images = []
    for batch, _ in ds:
        images.append(batch.numpy())
        if sum(len(b) for b in images) >= limit:
            break
    return np.concatenate(images)[:limit]


def main(argv=None):
    args = parse_args(argv)
    keras.utils.set_random_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_labels(args.labels, args.images_dir, args.target_column)
    splits = split_by_patient(df, args.target_column, args.task, args.threshold,
                              args.val_fraction, args.test_fraction, args.seed)
    print(f"Loaded {len(df)} images from {df['patient_id'].nunique()} patients")
    split_summary = summarise_splits(splits, args.task)
    if df["patient_id"].nunique() < MIN_RECOMMENDED_PATIENTS:
        print(f"WARNING: fewer than {MIN_RECOMMENDED_PATIENTS} patients; "
              "test metrics will have wide confidence intervals.")

    ds_kwargs = dict(batch_size=args.batch_size, image_size=args.image_size, crop_fraction=args.crop_fraction)
    train_ds = make_dataset(splits["train"], training=True, seed=args.seed, **ds_kwargs)
    val_ds = make_dataset(splits["val"], **ds_kwargs)
    test_ds = make_dataset(splits["test"], **ds_kwargs)

    target_mean = float(splits["train"].targets.mean())
    target_std = float(splits["train"].targets.std() or 1.0)
    model, base = build_model(args.task, args.image_size, args.backbone,
                              None if args.weights == "none" else args.weights,
                              target_mean, target_std)
    weights = class_weights(splits["train"].labels) if args.task == TASK_CLASSIFY else None
    checkpoint = out / "best.keras"

    print("Phase 1: training the head with the backbone frozen")
    compile_model(model, args.task, args.lr_head)
    fit(model, train_ds, val_ds, args.epochs_head, args.task, weights, checkpoint)

    print(f"Phase 2: fine-tuning the top {args.finetune_layers} backbone layers")
    unfreeze_top(base, args.finetune_layers)
    compile_model(model, args.task, args.lr_finetune)
    fit(model, train_ds, val_ds, args.epochs_finetune, args.task, weights, checkpoint)
    model.save(out / "model.keras")

    val_pred = model.predict(val_ds, verbose=0).reshape(-1)
    test_pred = model.predict(test_ds, verbose=0).reshape(-1)
    if args.task == TASK_CLASSIFY:
        decision_threshold = choose_threshold(splits["val"].labels, val_pred)
        test_metrics = classification_metrics(splits["test"].labels, test_pred, decision_threshold)
        meets_target = test_metrics["accuracy"] >= args.target_accuracy
        output_spec = {
            "type": "probability",
            "threshold": decision_threshold,
            "positive_label": f"{args.target_column} ≥ {args.threshold:g}{args.unit}",
            "negative_label": f"{args.target_column} < {args.threshold:g}{args.unit}",
        }
    else:
        test_metrics = regression_metrics(splits["test"].targets, test_pred)
        meets_target = test_metrics["meets_iso15197"]
        output_spec = {"type": "value", "unit": args.unit}
    output_spec["target_column"] = args.target_column
    output_spec["target_threshold"] = args.threshold if args.task == TASK_CLASSIFY else None

    tflite_path = out / MODEL_FILENAME
    export_tflite(model, tflite_path)
    tolerance = 0.02 if args.task == TASK_CLASSIFY else 0.02 * max(target_std, 1.0)
    parity = check_parity(model, tflite_path, collect_images(test_ds, 16), tolerance)

    metadata = {
        "schema_version": 1,
        "model_file": MODEL_FILENAME,
        "task": args.task,
        "backbone": args.backbone,
        "input": {"size": args.image_size, "crop_fraction": args.crop_fraction,
                  "channels": "RGB", "dtype": "float32", "range": [0, 255]},
        "output": output_spec,
        "test_metrics": test_metrics,
        "target_accuracy": args.target_accuracy if args.task == TASK_CLASSIFY else None,
        "meets_target": bool(meets_target),
        "dataset": split_summary,
        "synthetic_data": args.synthetic_data,
        "clinically_validated": False,
        "tflite_max_abs_diff": parity,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_metadata(out / METADATA_FILENAME, metadata)
    (out / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2) + "\n")

    print("\nHeld-out test results (patients never seen in training):")
    print(json.dumps(test_metrics, indent=2))
    if args.task == TASK_CLASSIFY:
        verdict = "MET" if meets_target else "NOT MET"
        low, high = test_metrics["accuracy_ci95"] or (float("nan"), float("nan"))
        print(f"Target accuracy {args.target_accuracy:.0%}: {verdict} "
              f"(got {test_metrics['accuracy']:.1%}, 95% CI {low:.0%}-{high:.0%}, "
              f"majority baseline {test_metrics['majority_baseline_accuracy']:.1%})")
    else:
        verdict = "MET" if meets_target else "NOT MET"
        print(f"ISO 15197 accuracy (>=95% in zone): {verdict} "
              f"({test_metrics['within_iso15197_percent']:.1f}% in zone)")

    if args.install_to_app:
        assets = install_to_app(out, args.install_to_app)
        print(f"Installed model into {assets}")
    return metadata


if __name__ == "__main__":
    main()
