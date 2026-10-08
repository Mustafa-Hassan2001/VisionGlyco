// Must match ml/visionglyco/preprocess.py: centre-crop a square of
// crop_fraction x the shorter side, resize to size x size, RGB float32 in 0-255.

export type CropRect = { originX: number; originY: number; width: number; height: number };

export function centerCropRect(width: number, height: number, cropFraction: number): CropRect {
  const side = Math.round(Math.min(width, height) * cropFraction);
  return {
    originX: Math.floor((width - side) / 2),
    originY: Math.floor((height - side) / 2),
    width: side,
    height: side,
  };
}

/** Convert decoded RGBA pixels into the model's NHWC float32 RGB input. */
export function rgbaToModelInput(rgba: Uint8Array, width: number, height: number): Float32Array {
  const pixels = width * height;
  if (rgba.length !== pixels * 4) {
    throw new Error(`Expected ${pixels * 4} RGBA bytes, got ${rgba.length}`);
  }
  const out = new Float32Array(pixels * 3);
  for (let i = 0; i < pixels; i++) {
    out[i * 3] = rgba[i * 4];
    out[i * 3 + 1] = rgba[i * 4 + 1];
    out[i * 3 + 2] = rgba[i * 4 + 2];
  }
  return out;
}
