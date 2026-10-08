import type { ModelMetadata, Prediction } from './types';

export function interpretOutput(raw: number, metadata: ModelMetadata): Prediction {
  if (!Number.isFinite(raw)) {
    throw new Error('Model returned a non-numeric result');
  }
  if (metadata.task === 'classify') {
    const probability = Math.min(1, Math.max(0, raw));
    const positive = probability >= metadata.output.threshold;
    return {
      kind: 'probability',
      probability,
      positive,
      label: positive ? metadata.output.positive_label : metadata.output.negative_label,
    };
  }
  const { unit } = metadata.output;
  return { kind: 'value', value: raw, unit, label: `${raw.toFixed(1)} ${unit}` };
}

/** Warnings for models that are not ready to show (synthetic test models, missed accuracy target). */
export function modelWarnings(metadata: ModelMetadata): string[] {
  const warnings: string[] = [];
  if (metadata.synthetic_data) {
    warnings.push('This model was trained on SYNTHETIC test images. Its results are meaningless.');
  }
  if (!metadata.meets_target) {
    warnings.push(
      metadata.task === 'classify'
        ? `Held-out accuracy is below the ${Math.round((metadata.target_accuracy ?? 0) * 100)}% target.`
        : 'Held-out accuracy does not meet the ISO 15197 glucose-meter standard.',
    );
  }
  return warnings;
}
