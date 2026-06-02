export type TryOnModelId =
  | 'google/gemini-3.1-flash-image-preview'
  | 'google/gemini-2.5-flash-image'
  | 'openai/gpt-5.4-image-2'
  | 'recraft/recraft-v4.1-utility-pro';

export type TryOnModel = {
  id: TryOnModelId;
  name: string;
  shortName: string;
  provider: string;
  tier: 'value' | 'balanced' | 'quality' | 'experimental';
  costLabel: string;
  latencyLabel: string;
  description: string;
  maxInputImages: 1 | 2;
  referenceMode: 'native-two-image' | 'composite-reference';
  strengths: string[];
  caveat?: string;
};

export const TRY_ON_MODELS: TryOnModel[] = [
  {
    id: 'google/gemini-3.1-flash-image-preview',
    name: 'Google: Nano Banana 2',
    shortName: 'Balanced',
    provider: 'Google',
    tier: 'balanced',
    costLabel: 'Lower cost',
    latencyLabel: 'Fast',
    description: 'Newer Flash image model with native two-reference support.',
    maxInputImages: 2,
    referenceMode: 'native-two-image',
    strengths: ['Native person + garment references', 'Good iteration cost', 'Fast previews'],
  },
  {
    id: 'google/gemini-2.5-flash-image',
    name: 'Google: Nano Banana',
    shortName: 'Value',
    provider: 'Google',
    tier: 'value',
    costLabel: 'Lowest native cost',
    latencyLabel: 'Fast',
    description: 'The cheapest native two-image option in the current OpenRouter image catalog.',
    maxInputImages: 2,
    referenceMode: 'native-two-image',
    strengths: ['Very low cost', 'Native two-image editing', 'Good retry model'],
  },
  {
    id: 'openai/gpt-5.4-image-2',
    name: 'OpenAI: GPT-5.4 Image 2',
    shortName: 'Quality',
    provider: 'OpenAI',
    tier: 'quality',
    costLabel: 'Premium',
    latencyLabel: 'Slower',
    description: 'Highest-fidelity OpenAI image model currently listed by OpenRouter.',
    maxInputImages: 2,
    referenceMode: 'native-two-image',
    strengths: ['Strong instruction following', 'High detail preservation', 'Best for final renders'],
  },
  {
    id: 'recraft/recraft-v4.1-utility-pro',
    name: 'Recraft: V4.1 Utility Pro',
    shortName: 'Recraft',
    provider: 'Recraft',
    tier: 'experimental',
    costLabel: 'Per-image',
    latencyLabel: 'Medium',
    description: 'Design-first image model tested through a composite reference board.',
    maxInputImages: 1,
    referenceMode: 'composite-reference',
    strengths: ['Product-style restraint', 'Natural photorealism', 'Useful for model comparison'],
    caveat: 'OpenRouter notes Recraft accepts only one input image, so this app sends a combined reference board.',
  },
];

export const DEFAULT_TRY_ON_MODEL_ID: TryOnModelId = 'google/gemini-3.1-flash-image-preview';

export function findTryOnModel(modelId: string | undefined | null) {
  if (!modelId) {
    return undefined;
  }

  return TRY_ON_MODELS.find((model) => model.id === modelId);
}

export function getDefaultTryOnModel(envModelId?: string) {
  return findTryOnModel(envModelId) || findTryOnModel(DEFAULT_TRY_ON_MODEL_ID)!;
}

export function getPublicModelOptions() {
  return TRY_ON_MODELS.map((model) => ({
    id: model.id,
    name: model.name,
    shortName: model.shortName,
    provider: model.provider,
    tier: model.tier,
    costLabel: model.costLabel,
    latencyLabel: model.latencyLabel,
    description: model.description,
    referenceMode: model.referenceMode,
    strengths: model.strengths,
    caveat: model.caveat,
  }));
}
