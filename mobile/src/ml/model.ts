import { loadTensorflowModel, type TfliteModel } from 'react-native-fast-tflite';

import { bundledMetadata, bundledModel } from './bundledModel';
import type { ModelMetadata } from './types';

export type LoadedModel = { tflite: TfliteModel; metadata: ModelMetadata };

let loading: Promise<LoadedModel> | null = null;

export function isModelInstalled(): boolean {
  return bundledModel != null && bundledMetadata != null;
}

export function getMetadata(): ModelMetadata | null {
  return bundledMetadata;
}

/** Load the bundled model once and check it matches its metadata. */
export function loadModel(): Promise<LoadedModel> {
  if (!loading) {
    loading = load().catch((error) => {
      loading = null;
      throw error;
    });
  }
  return loading;
}

async function load(): Promise<LoadedModel> {
  if (bundledModel == null || bundledMetadata == null) {
    throw new Error('No model is installed. Train one with ml/train.py --install-to-app.');
  }
  const tflite = await loadTensorflowModel(bundledModel, []);
  const { size, polar } = bundledMetadata.input;
  const input = tflite.inputs[0];
  const expected = polar ? [1, polar.height, polar.width, 3] : [1, size, size, 3];
  if (
    input == null ||
    input.dataType !== 'float32' ||
    input.shape.length !== 4 ||
    input.shape.some((dim, i) => dim !== expected[i])
  ) {
    throw new Error(
      `Model input ${JSON.stringify(input?.shape)} ${input?.dataType} does not match ` +
        `metadata ${JSON.stringify(expected)} float32`,
    );
  }
  if (tflite.outputs[0]?.dataType !== 'float32') {
    throw new Error('Model output must be float32');
  }
  return { tflite, metadata: bundledMetadata };
}

export async function runModel(model: LoadedModel, input: Float32Array): Promise<number> {
  const buffer = input.buffer.slice(input.byteOffset, input.byteOffset + input.byteLength);
  const [output] = await model.tflite.run([buffer as ArrayBuffer]);
  if (output == null) {
    throw new Error('Model produced no output');
  }
  return new Float32Array(output)[0];
}
