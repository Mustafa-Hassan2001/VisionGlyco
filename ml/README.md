# VisionGlyco model training

Transfer-learning pipeline that trains an eye-image model, measures it on held-out
patients, and exports a TFLite file for the React Native app in `../mobile`.

## 1. Prepare labelled data

The model can only be as good as its labels. Each photo needs a **real measurement
taken at the same visit**. Random or made-up labels produce a model that learns nothing.

```
data/
  images/
    p0001_left.jpg
    p0001_right.jpg
    ...
  labels.csv
```

`labels.csv`:

| image | patient_id | hba1c |
|---|---|---|
| p0001_left.jpg | P0001 | 7.2 |
| p0001_right.jpg | P0001 | 7.2 |
| p0002_left.jpg | P0002 | 5.4 |

- `image`: path relative to `--images-dir`
- `patient_id`: used to keep each person in only one of train/validation/test
- target column: `hba1c` (%) by default; any numeric column works via `--target-column`

Capture every image the same way: same adapter, lighting and distance, with the eye
centred. The pipeline keeps the central 80% square, the same region the app's guide
circle shows.

## 2. Install and train

```bash
cd ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Recommended: screening classifier, HbA1c >= 6.5% vs < 6.5%
python train.py --labels data/labels.csv --images-dir data/images \
    --task classify --target-column hba1c --threshold 6.5 \
    --install-to-app ../mobile
```

Training runs in two phases: the ImageNet-pretrained EfficientNet head first, then
fine-tuning of the top backbone layers. A GPU helps but is not required for about
1,500 images.

Regression (predicting a number) is also supported, though no published evidence
supports it from the iris:

```bash
python train.py --labels data/labels.csv --images-dir data/images \
    --task regress --target-column glucose_mg_dl --unit mg/dL
```

## 3. Read the results honestly

`outputs/` contains:

| File | Contents |
|---|---|
| `visionglyco.tflite` | float16 model for the phone |
| `model_metadata.json` | preprocessing, decision threshold, test metrics (read by the app) |
| `test_metrics.json` | held-out metrics only |
| `model.keras`, `best.keras` | Keras checkpoints |

What the reported numbers mean:

- **Patient-level split.** No patient's photos appear in more than one of train,
  validation and test. Mixing them inflates accuracy because photos of the same eye
  are near-duplicates.
- **Threshold chosen on validation, reported on test.** The test set is never used
  to tune anything.
- **Baselines.** `majority_baseline_accuracy` is what always guessing the most common
  class scores. If 80% of your patients are diabetic, 80% accuracy means the model
  learned nothing. Compare balanced accuracy and AUC as well.
- **95% confidence intervals.** With 30 test patients, an 85% accuracy may really be
  anywhere from 70% to 95%.
- **`--target-accuracy 0.85`** only *checks* whether the target was met. The app
  shows a warning when it wasn't. No setting can force a particular accuracy; the
  result depends on whether the images contain the signal.

To check that everything runs without real data, train on drawn synthetic
"eyes". The app labels such a model as synthetic and meaningless:

```bash
python tools/make_synthetic_dataset.py --out data/synthetic
python train.py --labels data/synthetic/labels.csv --images-dir data/synthetic/images \
    --synthetic-data --install-to-app ../mobile
```

## 4. Predict from the command line

```bash
python predict.py --model-dir outputs photo.jpg
```

## Tests

```bash
pytest            # includes a ~1 minute end-to-end train/export/predict run
pytest -m "not slow"
```

## Before any clinical use

This is research software. Showing results to patients requires an ethics-approved
validation study against laboratory reference measurements, ideally at a different
hospital from the training data, and regulatory clearance as Software as a Medical
Device (FDA, EU MDR, or DRAP in Pakistan).
