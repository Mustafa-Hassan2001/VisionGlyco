import { describe, expect, it } from '@jest/globals';

import { estimatePupilRadius, unwrapIris } from '../polar';
import fixture from './fixtures/polar_fixture.json';

const limits = { pupil_min: 0.15, pupil_max: 0.7, pupil_default: 0.35 };

/** Square RGBA image: pupil colour inside pupilFrac, then `ring(x, y)` out to the edge. */
function eye(size: number, pupilFrac: number, ring: (x: number, y: number) => number[]) {
  const c = (size - 1) / 2;
  const data = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const rel = Math.hypot(x - c, y - c) / (size / 2);
      const [r, g, b] = rel < pupilFrac ? [10, 10, 10] : ring(x, y);
      data.set([r, g, b, 255], (y * size + x) * 4);
    }
  }
  return data;
}

describe('polar unwrap (matches ml/visionglyco/iris_polar.py)', () => {
  it('reproduces the Python reference output exactly', () => {
    const rgba = Uint8Array.from(fixture.rgba);
    const pupil = estimatePupilRadius(rgba, fixture.size, limits);
    expect(pupil).toBeCloseTo(fixture.pupil, 10);
    const strip = unwrapIris(rgba, fixture.size, pupil, fixture.height, fixture.width);
    expect(Array.from(strip)).toEqual(fixture.strip);
  });

  it('finds the pupil edge', () => {
    expect(estimatePupilRadius(eye(200, 0.3, () => [150, 120, 90]), 200, limits)).toBeCloseTo(0.3, 1);
  });

  it('falls back to the default pupil size on a featureless image', () => {
    expect(estimatePupilRadius(new Uint8Array(64 * 64 * 4).fill(128), 64, limits)).toBe(0.35);
  });

  it('starts at the right, turns towards the bottom, and runs pupil edge -> iris edge', () => {
    const size = 100;
    const c = (size - 1) / 2;
    // Right half blue, left half red; bottom quarter green near the iris edge.
    const img = eye(size, 0.2, (x, y) => (y > c + 40 ? [0, 255, 0] : x > c ? [0, 0, 255] : [255, 0, 0]));
    const h = 11;
    const w = 36;
    const strip = unwrapIris(img, size, 0.2, h, w);
    const px = (i: number, j: number) => Array.from(strip.slice((i * w + j) * 3, (i * w + j) * 3 + 3));
    // Row 0 sits exactly on the pupil edge, so check direction one row further out.
    expect(px(1, 0)).toEqual([0, 0, 255]); // 0 degrees: right
    expect(px(1, w / 2)).toEqual([255, 0, 0]); // 180 degrees: left
    expect(px(h - 1, w / 4)).toEqual([0, 255, 0]); // 90 degrees, outer edge: bottom
    expect(px(1, w / 4)).not.toEqual([0, 255, 0]); // 90 degrees, near pupil: not yet bottom band
  });
});
