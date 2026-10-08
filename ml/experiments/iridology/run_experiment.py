"""Diabetic vs non-diabetic classification on the public Farabi Hospital iris dataset.

Dataset: Moradi, Nazer, Khasahmadi, Mohammadzadeh & Khojasteh Jafari, "Discovering
Informative Regions in Iris Images to Predict Diabetes", ICBME 2018.
https://github.com/NaghmeNazer/diabetes-iridology  (no licence stated; cite if used)

    git clone --depth 1 https://github.com/NaghmeNazer/diabetes-iridology /tmp/iridology
    python experiments/iridology/run_experiment.py --data-zip /tmp/iridology/Data.zip

Uses the person-based split (one unwrapped iris strip per person, 88 diabetic / 108
control). Because there is one image per person, ordinary cross-validation cannot
leak a person between train and test.
"""

import argparse
import io
import json
import pickle
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SPLIT = "Data/personBase"
SOURCE = "https://github.com/NaghmeNazer/diabetes-iridology"

# The dataset ships as pickled numpy arrays. Unpickling untrusted data can run code,
# so only the classes needed to rebuild a numpy array are allowed.
_ALLOWED = {("numpy", "dtype"), ("numpy", "ndarray"), ("numpy.core.multiarray", "_reconstruct")}


class _NumpyOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in _ALLOWED:
            raise pickle.UnpicklingError(f"Refusing to load {module}.{name}")
        if module == "numpy.core.multiarray":
            from numpy._core import multiarray
            return getattr(multiarray, name)
        return getattr(np, name)


def load_dataset(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        def read(name):
            return _NumpyOnlyUnpickler(io.BytesIO(z.read(f"{SPLIT}/{name}.p")), encoding="latin1").load()
        diabetic = read("diabeteImageArr")[..., ::-1]  # stored BGR -> RGB
        control = read("controlImageArr")[..., ::-1]
    images = np.concatenate([diabetic, control]).astype(np.float32)
    labels = np.array([1] * len(diabetic) + [0] * len(control))
    return images, labels


def normalise_per_image(images):
    """Remove each image's colour cast and brightness/contrast."""
    z = (images - images.mean(axis=(1, 2), keepdims=True)) / (images.std(axis=(1, 2), keepdims=True) + 1e-6)
    return np.clip(z * 40 + 128, 0, 255)


def make_classifier(c=0.1):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))


def cross_validate(features, labels, repeats, folds, c=0.1):
    acc, bacc, auc, sens, spec = [], [], [], [], []
    for seed in range(repeats):
        cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
        prob = cross_val_predict(make_classifier(c), features, labels, cv=cv, method="predict_proba")[:, 1]
        pred = prob >= 0.5
        acc.append(accuracy_score(labels, pred))
        bacc.append(balanced_accuracy_score(labels, pred))
        auc.append(roc_auc_score(labels, prob))
        sens.append(pred[labels == 1].mean())
        spec.append((~pred)[labels == 0].mean())

    def summary(v):
        return {"mean": float(np.mean(v)), "std": float(np.std(v)),
                "min": float(np.min(v)), "max": float(np.max(v))}
    return {"accuracy": summary(acc), "balanced_accuracy": summary(bacc), "auc": summary(auc),
            "sensitivity": summary(sens), "specificity": summary(spec)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-zip", required=True, help="Data.zip from the diabetes-iridology repository")
    p.add_argument("--out", default="outputs/iridology")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--folds", type=int, default=5)
    args = p.parse_args(argv)

    import keras  # imported late so --help is fast

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    images, labels = load_dataset(args.data_zip)
    n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
    print(f"{len(labels)} people: {n_pos} diabetic, {n_neg} control; image size {images.shape[1:]}")

    backbone = keras.applications.EfficientNetB0(include_top=False, weights="imagenet",
                                                 input_shape=images.shape[1:], pooling="avg")
    deep_raw = backbone.predict(images, batch_size=8, verbose=0)
    deep_norm = backbone.predict(normalise_per_image(images), batch_size=8, verbose=0)

    experiments = {
        "efficientnet_b0": ("EfficientNet-B0 features + logistic regression", deep_raw),
        "efficientnet_b0_normalised": ("Same, colour/brightness normalised per image", deep_norm),
    }
    results = {}
    print(f"\n{args.repeats}x repeated {args.folds}-fold cross-validation")
    print(f"{'Model':52s} {'Accuracy':>16s} {'AUC':>7s}")
    print(f"{'Always predict the larger class':52s} {max(n_pos, n_neg) / len(labels):>15.1%}  {0.5:>6.3f}")
    for key, (name, feats) in experiments.items():
        r = cross_validate(feats, labels, args.repeats, args.folds)
        results[key] = {"description": name, **r}
        print(f"{name:52s} {r['accuracy']['mean']:>8.1%} ± {r['accuracy']['std']:.1%}  {r['auc']['mean']:>6.3f}")

    # Final model: fitted on all 196 people (its accuracy is the cross-validated figure above).
    final = make_classifier().fit(deep_raw, labels)
    scaler, logreg = final.named_steps["standardscaler"], final.named_steps["logisticregression"]
    model = {
        "description": "Logistic regression on ImageNet EfficientNet-B0 pooled features "
                       "(keras.applications, include_top=False, pooling='avg').",
        "input": {"format": "unwrapped (polar) iris strip as in the source dataset",
                  "height": int(images.shape[1]), "width": int(images.shape[2]),
                  "channels": "RGB", "range": [0, 255]},
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": logreg.coef_[0].tolist(),
        "intercept": float(logreg.intercept_[0]),
        "positive_label": "diabetic",
    }
    (out / "model.json").write_text(json.dumps(model) + "\n")

    report = {
        "dataset": {"source": SOURCE, "split": SPLIT, "people": len(labels),
                    "diabetic": n_pos, "control": n_neg, "image_shape": list(images.shape[1:])},
        "evaluation": f"{args.repeats}x repeated stratified {args.folds}-fold cross-validation",
        "majority_baseline_accuracy": max(n_pos, n_neg) / len(labels),
        "results": results,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nSaved {out / 'results.json'} and {out / 'model.json'}")
    return report


if __name__ == "__main__":
    main()
