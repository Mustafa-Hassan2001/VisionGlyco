import type { Prediction } from '../ml/types';

export function formatPercent(value: number | null | undefined, digits = 1): string {
  return value == null ? '—' : `${(value * 100).toFixed(digits)}%`;
}

export function formatCi(ci: [number, number] | null | undefined): string {
  return ci == null ? '' : ` (95% CI ${formatPercent(ci[0], 0)}–${formatPercent(ci[1], 0)})`;
}

export function formatDate(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  });
}

export function predictionHeadline(p: Prediction): string {
  return p.kind === 'probability' ? p.label : `${p.value.toFixed(1)} ${p.unit}`;
}

export function predictionDetail(p: Prediction): string {
  return p.kind === 'probability'
    ? `Model score ${formatPercent(p.probability, 0)}`
    : 'Model estimate';
}

/** Numeric value used for the trend chart. */
export function predictionValue(p: Prediction): number {
  return p.kind === 'probability' ? p.probability : p.value;
}
