// Mirrors the model_metadata.json written by ml/train.py.

export type ClassificationMetrics = {
  n: number;
  n_positive: number;
  threshold: number;
  accuracy: number;
  accuracy_ci95: [number, number] | null;
  balanced_accuracy: number | null;
  auc: number | null;
  auc_ci95?: [number, number] | null;
  sensitivity: number | null;
  specificity: number | null;
  majority_baseline_accuracy: number;
};

export type RegressionMetrics = {
  n: number;
  mae: number;
  mard_percent: number;
  within_iso15197_percent: number;
  meets_iso15197: boolean;
  mean_baseline_mae: number;
};

type SplitSummary = { images: number; patients: number; positive_images?: number };

type CommonMetadata = {
  schema_version: 1;
  model_file: string;
  backbone: string;
  input: {
    size: number;
    crop_fraction: number;
    channels: 'RGB';
    dtype: 'float32';
    range: [number, number];
    // Present when the model takes an unwrapped iris strip instead of a square image.
    polar?: {
      height: number;
      width: number;
      pupil_min: number;
      pupil_max: number;
      pupil_default: number;
    };
  };
  target_accuracy: number | null;
  meets_target: boolean;
  dataset: { train: SplitSummary; val: SplitSummary; test: SplitSummary };
  synthetic_data: boolean;
  clinically_validated: boolean;
  trained_at: string;
  evaluation?: string;
  source?: { name: string; citation: string; url: string };
};

export type ModelMetadata =
  | (CommonMetadata & {
      task: 'classify';
      output: {
        type: 'probability';
        threshold: number;
        positive_label: string;
        negative_label: string;
        target_column: string;
        target_threshold: number;
      };
      test_metrics: ClassificationMetrics;
    })
  | (CommonMetadata & {
      task: 'regress';
      output: { type: 'value'; unit: string; target_column: string };
      test_metrics: RegressionMetrics;
    });

export type Prediction =
  | { kind: 'probability'; probability: number; positive: boolean; label: string }
  | { kind: 'value'; value: number; unit: string; label: string };

export type ScanRecord = {
  id: string;
  createdAt: string;
  imageUri: string;
  prediction: Prediction;
  modelTrainedAt: string;
};
