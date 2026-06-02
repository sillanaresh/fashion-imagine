'use client';

import { BadgeCheck, Gauge, Zap } from 'lucide-react';
import { TRY_ON_MODELS, type TryOnModelId } from '@/lib/model-catalog';

type ModelSelectorProps = {
  selectedModelId: TryOnModelId;
  onModelChange: (modelId: TryOnModelId) => void;
};

const tierIcons = {
  value: Zap,
  balanced: Gauge,
  quality: BadgeCheck,
};

export default function ModelSelector({ selectedModelId, onModelChange }: ModelSelectorProps) {
  return (
    <fieldset className="model-selector">
      <legend>
        <span className="ui-eyebrow">Model</span>
        Generation route
      </legend>
      <div className="model-options">
        {TRY_ON_MODELS.map((model) => {
          const Icon = tierIcons[model.tier];
          return (
            <label key={model.id} className={selectedModelId === model.id ? 'is-selected' : ''}>
              <input
                type="radio"
                name="modelId"
                value={model.id}
                checked={selectedModelId === model.id}
                onChange={() => onModelChange(model.id)}
              />
              <span className="model-option__top">
                <span>
                  <Icon size={17} strokeWidth={1.8} />
                  {model.shortName}
                </span>
                <small>{model.costLabel}</small>
              </span>
              <strong>{model.name}</strong>
              <span className="model-option__description">{model.description}</span>
            </label>
          );
        })}
      </div>
    </fieldset>
  );
}
