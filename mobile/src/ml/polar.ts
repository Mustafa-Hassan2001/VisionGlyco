// Iris unwrapping, mirrored from ml/visionglyco/iris_polar.py (kept identical so the
// phone feeds the model the same strips it was trained on; see polar.test.ts).
//
// Input: square RGBA pixels whose inscribed circle is the iris. Output: float32 RGB
// strip, height x width. Column j is the ray at j * 360 / width degrees, starting at
// the right and turning towards the bottom; row 0 is the pupil edge, the last row the
// outer iris edge.

const RING_BINS = 50;
const IRIS_EDGE = 0.98;
const MIN_CONTRAST = 10;
const PUPIL_EDGE_FACTOR = 0.1;

export type PupilLimits = { pupil_min: number; pupil_max: number; pupil_default: number };

function median(values: number[]): number {
  const s = [...values].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

/** Pupil radius as a fraction of the iris radius. */
export function estimatePupilRadius(rgba: Uint8Array, size: number, limits: PupilLimits): number {
  const centre = (size - 1) / 2;
  const radius = size / 2;
  const sums = new Float64Array(RING_BINS);
  const counts = new Float64Array(RING_BINS);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const bin = Math.floor((Math.hypot(x - centre, y - centre) / radius) * RING_BINS);
      if (bin >= RING_BINS) continue;
      const i = (y * size + x) * 4;
      sums[bin] += 0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2];
      counts[bin] += 1;
    }
  }
  const means = Array.from(sums, (s, k) => (counts[k] > 0 ? s / counts[k] : NaN));
  const centreOf = (k: number) => (k + 0.5) / RING_BINS;

  let darkest = -1;
  const outer: number[] = [];
  for (let k = 0; k < RING_BINS; k++) {
    if (counts[k] === 0) continue;
    if (centreOf(k) < 0.6 && (darkest < 0 || means[k] < means[darkest])) darkest = k;
    if (centreOf(k) >= 0.6 && centreOf(k) < 0.9) outer.push(means[k]);
  }
  if (darkest < 0 || outer.length === 0) return limits.pupil_default;
  const irisLevel = median(outer);
  if (irisLevel - means[darkest] < MIN_CONTRAST) return limits.pupil_default;

  const threshold = means[darkest] + PUPIL_EDGE_FACTOR * (irisLevel - means[darkest]);
  for (let k = darkest; k < RING_BINS; k++) {
    if (counts[k] > 0 && means[k] > threshold) {
      return Math.min(limits.pupil_max, Math.max(limits.pupil_min, k / RING_BINS));
    }
  }
  return limits.pupil_default;
}

/** Nearest-neighbour polar unwrap into an NHWC float32 RGB strip (0-255). */
export function unwrapIris(
  rgba: Uint8Array,
  size: number,
  pupilFraction: number,
  height: number,
  width: number,
): Float32Array {
  const centre = (size - 1) / 2;
  const radius = size / 2;
  const out = new Float32Array(height * width * 3);
  const clamp = (v: number) => Math.min(size - 1, Math.max(0, v));
  for (let j = 0; j < width; j++) {
    const theta = ((j * 360) / width) * (Math.PI / 180);
    const cos = Math.cos(theta);
    const sin = Math.sin(theta);
    for (let i = 0; i < height; i++) {
      const r = (pupilFraction + (i / (height - 1)) * (IRIS_EDGE - pupilFraction)) * radius;
      const x = clamp(Math.floor(centre + r * cos + 0.5));
      const y = clamp(Math.floor(centre + r * sin + 0.5));
      const src = (y * size + x) * 4;
      const dst = (i * width + j) * 3;
      out[dst] = rgba[src];
      out[dst + 1] = rgba[src + 1];
      out[dst + 2] = rgba[src + 2];
    }
  }
  return out;
}
