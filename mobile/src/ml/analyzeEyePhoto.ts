import { File } from 'expo-file-system';
import { ImageManipulator, SaveFormat } from 'expo-image-manipulator';
import * as jpeg from 'jpeg-js';

import { interpretOutput } from './interpret';
import { loadModel, runModel } from './model';
import { estimatePupilRadius, unwrapIris } from './polar';
import { centerCropRect, rgbaToModelInput } from './preprocess';
import type { Prediction } from './types';

const THUMBNAIL_SIZE = 480;

export type Analysis = { prediction: Prediction; thumbnailUri: string };

export type AnalyzeOptions = {
  /** The user already cropped the photo to the iris (gallery upload), so keep all of it. */
  croppedToIris?: boolean;
};

/** Photo URI -> centre crop -> model input -> on-device prediction. */
export async function analyzeEyePhoto(photoUri: string, options: AnalyzeOptions = {}): Promise<Analysis> {
  const model = await loadModel();
  const { size, crop_fraction: guideFraction, polar } = model.metadata.input;
  const cropFraction = options.croppedToIris ? 1 : guideFraction;

  // Rendering applies EXIF orientation, so width/height are as displayed.
  const original = await ImageManipulator.manipulate(photoUri).renderAsync();
  const rect = centerCropRect(original.width, original.height, cropFraction);

  const modelImage = await ImageManipulator.manipulate(photoUri)
    .crop(rect)
    .resize({ width: size, height: size })
    .renderAsync();
  const saved = await modelImage.saveAsync({ format: SaveFormat.JPEG, compress: 1 });
  const decoded = jpeg.decode(await new File(saved.uri).bytes(), {
    useTArray: true,
    formatAsRGBA: true,
  });
  if (decoded.width !== size || decoded.height !== size) {
    throw new Error(`Resized image is ${decoded.width}x${decoded.height}, expected ${size}x${size}`);
  }

  // Iris models take an unwrapped strip from pupil edge to iris edge.
  const input = polar
    ? unwrapIris(decoded.data, size, estimatePupilRadius(decoded.data, size, polar), polar.height, polar.width)
    : rgbaToModelInput(decoded.data, size, size);
  const raw = await runModel(model, input);

  const thumbnail = await ImageManipulator.manipulate(photoUri)
    .crop(rect)
    .resize({ width: THUMBNAIL_SIZE, height: THUMBNAIL_SIZE })
    .renderAsync();
  const thumb = await thumbnail.saveAsync({ format: SaveFormat.JPEG, compress: 0.85 });

  return { prediction: interpretOutput(raw, model.metadata), thumbnailUri: thumb.uri };
}
