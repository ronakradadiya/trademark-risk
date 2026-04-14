import * as path from 'node:path';
import * as fs from 'node:fs';
import * as ort from 'onnxruntime-node';
import { MLPredictionSchema, type MLPrediction } from '../schemas/index.js';

export const FEATURE_COLS = [
  'owner_filing_count_2yr',
  'owner_abandonment_rate',
  'owner_historical_cancellation_rate',
  'days_since_owner_first_filing',
  'owner_is_individual',
  'owner_is_foreign',
  'attorney_case_count',
  'attorney_cancellation_rate',
  'days_since_filing',
  'days_filing_to_registration',
  'was_abandoned',
  'is_currently_active',
  'class_breadth',
] as const;
export type FeatureName = (typeof FEATURE_COLS)[number];
export type FeatureVector = Record<FeatureName, number>;

export const HIGH_THRESHOLD = 0.7;
export const LOW_THRESHOLD = 0.08;
export const MODEL_VERSION = 'v4';
export const DEFAULT_MODELS_DIR = path.resolve(process.cwd(), '..', 'ml', 'models');

export class ClassifierError extends Error {
  constructor(message: string, public readonly code: 'missing_feature' | 'model_load' | 'inference') {
    super(message);
    this.name = 'ClassifierError';
  }
}

export interface Classifier {
  predict(features: FeatureVector): Promise<MLPrediction>;
}

function scoreToTier(score: number): MLPrediction['tier'] {
  if (score >= HIGH_THRESHOLD) return 'high';
  if (score < LOW_THRESHOLD) return 'low';
  return 'mid';
}

function featuresToRow(features: FeatureVector): Float32Array {
  const row = new Float32Array(FEATURE_COLS.length);
  for (let i = 0; i < FEATURE_COLS.length; i++) {
    const key = FEATURE_COLS[i]!;
    const v = features[key];
    if (v === undefined || v === null || Number.isNaN(v)) {
      throw new ClassifierError(`missing or invalid feature: ${key}`, 'missing_feature');
    }
    row[i] = v;
  }
  return row;
}

function extractProb(output: ort.InferenceSession.ReturnType): number {
  // XGBoost/LR ONNX (skl2onnx/onnxmltools) emit [label, [{0: p0, 1: p1}, ...]]
  // CatBoost emits a flat probability matrix [[p0, p1]]
  const entries = Object.values(output);
  for (const t of entries) {
    if (!t) continue;
    const data = (t as ort.Tensor).data;
    const dims = (t as ort.Tensor).dims;
    if (!data || !dims) continue;
    if (dims.length === 2 && dims[1] === 2) {
      return Number(data[1]);
    }
  }
  // Fallback: search for a Map output (ZipMap from onnxmltools)
  for (const t of entries) {
    const anyT = t as unknown as { data?: unknown };
    const data = anyT?.data;
    if (Array.isArray(data) && data.length > 0) {
      const first = data[0];
      if (first instanceof Map) {
        const v = first.get(1) ?? first.get('1');
        if (typeof v === 'number') return v;
      } else if (first && typeof first === 'object' && 1 in first) {
        return Number((first as Record<number, number>)[1]);
      }
    }
  }
  throw new ClassifierError('could not extract class-1 probability from ONNX output', 'inference');
}

export async function loadClassifier(modelsDir: string = DEFAULT_MODELS_DIR): Promise<Classifier> {
  const xgbPath = path.join(modelsDir, 'xgboost.onnx');
  const catPath = path.join(modelsDir, 'catboost.onnx');
  const metaPath = path.join(modelsDir, 'meta_lr.onnx');

  for (const p of [xgbPath, catPath, metaPath]) {
    if (!fs.existsSync(p)) {
      throw new ClassifierError(`model file not found: ${p}`, 'model_load');
    }
  }

  let xgb: ort.InferenceSession;
  let cat: ort.InferenceSession;
  let meta: ort.InferenceSession;
  try {
    [xgb, cat, meta] = await Promise.all([
      ort.InferenceSession.create(xgbPath),
      ort.InferenceSession.create(catPath),
      ort.InferenceSession.create(metaPath),
    ]);
  } catch (e) {
    throw new ClassifierError(
      `failed to load ONNX sessions: ${e instanceof Error ? e.message : String(e)}`,
      'model_load'
    );
  }

  const xgbInput = xgb.inputNames[0] ?? 'input';
  const catInput = cat.inputNames[0] ?? 'features';
  const metaInput = meta.inputNames[0] ?? 'meta_input';

  return {
    async predict(features) {
      const row = featuresToRow(features);
      const xTensor = new ort.Tensor('float32', row, [1, FEATURE_COLS.length]);

      let xgbProb: number;
      let catProb: number;
      try {
        const [xgbOut, catOut] = await Promise.all([
          xgb.run({ [xgbInput]: xTensor }),
          cat.run({ [catInput]: xTensor }),
        ]);
        xgbProb = extractProb(xgbOut);
        catProb = extractProb(catOut);
      } catch (e) {
        if (e instanceof ClassifierError) throw e;
        throw new ClassifierError(
          `base model inference failed: ${e instanceof Error ? e.message : String(e)}`,
          'inference'
        );
      }

      const metaData = new Float32Array([xgbProb, catProb]);
      const metaTensor = new ort.Tensor('float32', metaData, [1, 2]);

      let score: number;
      try {
        const metaOut = await meta.run({ [metaInput]: metaTensor });
        score = extractProb(metaOut);
      } catch (e) {
        throw new ClassifierError(
          `meta model inference failed: ${e instanceof Error ? e.message : String(e)}`,
          'inference'
        );
      }

      const clamped = Math.max(0, Math.min(1, score));
      return MLPredictionSchema.parse({
        score: clamped,
        tier: scoreToTier(clamped),
        high_threshold: HIGH_THRESHOLD,
        low_threshold: LOW_THRESHOLD,
        model_version: MODEL_VERSION,
      });
    },
  };
}
