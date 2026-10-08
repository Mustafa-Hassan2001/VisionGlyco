# Interim model: public iris dataset (Farabi Hospital)

The model currently bundled in the VisionGlyco app, trained while VisionGlyco's own
labelled data is being collected.

**Dataset:** [NaghmeNazer/diabetes-iridology](https://github.com/NaghmeNazer/diabetes-iridology).
196 people (88 diabetic, 108 non-diabetic), one unwrapped (polar) iris strip per person,
201×720 px, labelled diabetic / non-diabetic. The data is downloaded at run time, not
copied into this repository. Cite:

> P. Moradi, N. Nazer, A. Khasahmadi, H. Mohammadzadeh, H. Khojasteh Jafari.
> *Discovering Informative Regions in Iris Images to Predict Diabetes.* ICBME 2018.

No licence is stated for the dataset; ask the authors before a public release.

## Train and evaluate

```bash
git clone --depth 1 https://github.com/NaghmeNazer/diabetes-iridology /tmp/iridology
cd ml
python experiments/iridology/run_experiment.py --data-zip /tmp/iridology/Data.zip
```

The dataset is stored as pickle files; the script loads them with an unpickler that only
allows numpy arrays.

## Results (5× repeated 5-fold cross-validation)

| Model | Accuracy | AUC |
|---|---|---|
| Always predict "non-diabetic" (baseline) | 55.1% | 0.50 |
| **EfficientNet-B0 (ImageNet) features + logistic regression** | **90.9% ± 2.0%** | **0.951** |
| Same, colour/brightness normalised per image | 92.7% ± 1.1% | 0.949 |

Sensitivity 88.4%, specificity 93.0%. Full numbers are in `results/results.json`. The
final classifier, fitted on all 196 people, is in `results/model.json` (feature scaler +
logistic-regression weights).

## Install into the app

```bash
python experiments/iridology/export_app_model.py --data-zip /tmp/iridology/Data.zip --install-to-app ../mobile
```

This folds the logistic regression into the network as one sigmoid layer, exports a
TFLite model, and installs it into `mobile/`. In the app, each photo is cropped to the
iris, the pupil is located, and the iris is unwrapped into a 201×720 strip
(`ml/visionglyco/iris_polar.py`, mirrored in `mobile/src/ml/polar.ts`) before the model runs.

Export checks:

| Check | Result |
|---|---|
| Folded layer vs scikit-learn | max difference 3×10⁻⁷ |
| TFLite vs Keras | max difference 0.018 |
| Pupil estimate on dataset images drawn as round eyes | mean error 0.009 of the iris radius |

## Data collection for the VisionGlyco model

1. Photograph every participant with the same phone, adapter, lighting and settings.
2. Record age, sex and a same-day HbA1c for every participant.
3. Train with `ml/train.py` (patient-level train/validation/test split).
