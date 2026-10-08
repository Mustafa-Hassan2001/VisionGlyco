import { describe, expect, it } from '@jest/globals';

import { interpretOutput, modelWarnings } from '../interpret';
import { centerCropRect, rgbaToModelInput } from '../preprocess';
import type { ModelMetadata } from '../types';

const classifier: ModelMetadata = {
  schema_version: 1,
  model_file: 'visionglyco.tflite',
  backbone: 'efficientnet_b0',
  task: 'classify',
  input: { size: 2, crop_fraction: 0.8, channels: 'RGB', dtype: 'float32', range: [0, 255] },
  output: {
    type: 'probability',
    threshold: 0.4,
    positive_label: 'hba1c ≥ 6.5%',
    negative_label: 'hba1c < 6.5%',
    target_column: 'hba1c',
    target_threshold: 6.5,
  },
  test_metrics: {
    n: 10, n_positive: 5, threshold: 0.4, accuracy: 0.9, accuracy_ci95: [0.7, 1],
    balanced_accuracy: 0.9, auc: 0.95, sensitivity: 0.9, specificity: 0.9,
    majority_baseline_accuracy: 0.5,
  },
  target_accuracy: 0.85,
  meets_target: true,
  dataset: {
    train: { images: 70, patients: 30 },
    val: { images: 15, patients: 6 },
    test: { images: 15, patients: 6 },
  },
  synthetic_data: false,
  clinically_validated: false,
  trained_at: '2026-01-01T00:00:00+00:00',
};

describe('centerCropRect (matches ml/visionglyco/preprocess.py)', () => {
  it('crops a centred square of the shorter side', () => {
    expect(centerCropRect(4000, 3000, 0.8)).toEqual({ originX: 800, originY: 300, width: 2400, height: 2400 });
    expect(centerCropRect(3000, 4000, 0.8)).toEqual({ originX: 300, originY: 800, width: 2400, height: 2400 });
  });
});

describe('rgbaToModelInput', () => {
  it('drops alpha and keeps 0-255 values in NHWC order', () => {
    const rgba = new Uint8Array([1, 2, 3, 255, 4, 5, 6, 255, 7, 8, 9, 255, 10, 11, 12, 255]);
    expect(Array.from(rgbaToModelInput(rgba, 2, 2))).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
  });

  it('rejects buffers of the wrong size', () => {
    expect(() => rgbaToModelInput(new Uint8Array(3), 2, 2)).toThrow();
  });
});

describe('interpretOutput', () => {
  it('applies the validation-chosen threshold, not 0.5', () => {
    expect(interpretOutput(0.45, classifier)).toMatchObject({ positive: true, label: 'hba1c ≥ 6.5%' });
    expect(interpretOutput(0.35, classifier)).toMatchObject({ positive: false, label: 'hba1c < 6.5%' });
  });

  it('rejects NaN', () => {
    expect(() => interpretOutput(NaN, classifier)).toThrow();
  });
});

describe('modelWarnings', () => {
  it('shows no warnings for a model that meets its target', () => {
    expect(modelWarnings(classifier)).toEqual([]);
  });

  it('flags synthetic models and missed accuracy targets', () => {
    const w = modelWarnings({ ...classifier, synthetic_data: true, meets_target: false });
    expect(w.join(' ')).toMatch(/SYNTHETIC/);
    expect(w.join(' ')).toMatch(/below the 85% target/);
  });
});
