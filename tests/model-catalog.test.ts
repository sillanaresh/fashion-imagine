import { describe, expect, it } from 'vitest';
import {
  DEFAULT_TRY_ON_MODEL_ID,
  TRY_ON_MODELS,
  findTryOnModel,
  getDefaultTryOnModel,
} from '../lib/model-catalog';

describe('try-on model catalog', () => {
  it('keeps every model in the allowlist discoverable by id', () => {
    for (const model of TRY_ON_MODELS) {
      expect(findTryOnModel(model.id)).toEqual(model);
    }
  });

  it('falls back to the balanced default when an env model is unsupported', () => {
    expect(getDefaultTryOnModel('unknown/model').id).toBe(DEFAULT_TRY_ON_MODEL_ID);
  });

  it('marks one-input models as composite-reference routes', () => {
    const oneInputModels = TRY_ON_MODELS.filter((model) => model.maxInputImages === 1);
    expect(oneInputModels.length).toBeGreaterThan(0);
    expect(oneInputModels.every((model) => model.referenceMode === 'composite-reference')).toBe(true);
  });
});
