"""Shared constants. The mobile app reads the same values from model_metadata.json,
so preprocessing on the phone matches preprocessing during training."""

IMAGE_SIZE = 224
# Fraction of the shorter image side kept by the centre crop. The app's capture
# screen draws its alignment circle over the same region.
CROP_FRACTION = 0.8

TASK_CLASSIFY = "classify"
TASK_REGRESS = "regress"

# HbA1c >= 6.5% is the ADA diagnostic cut-off for diabetes.
DEFAULT_TARGET_COLUMN = "hba1c"
DEFAULT_THRESHOLD = 6.5

# ISO 15197:2013 accuracy zone: within +/-15 mg/dL below 100 mg/dL,
# within +/-15% at or above 100 mg/dL.
ISO_ABS_LIMIT_MG_DL = 15.0
ISO_REL_LIMIT = 0.15
ISO_SWITCH_MG_DL = 100.0
