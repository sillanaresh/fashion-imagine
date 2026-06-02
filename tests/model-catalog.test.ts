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

  it('only exposes native two-reference models for virtual try-on', () => {
    expect(TRY_ON_MODELS.every((model) => model.referenceMode === 'native-two-image')).toBe(true);
  });
});
