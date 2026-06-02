'use client';

import { useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { CheckCircle2, LockKeyhole, ShieldCheck, Sparkles, WandSparkles } from 'lucide-react';
import ImageUploader, { type UploadedImage } from '@/components/ImageUploader';
import ModelSelector from '@/components/ModelSelector';
import ResultDisplay from '@/components/ResultDisplay';
import {
  DEFAULT_TRY_ON_MODEL_ID,
  TRY_ON_MODELS,
  type TryOnModelId,
} from '@/lib/model-catalog';

type TryOnResponse = {
  resultImage?: string;
  error?: string;
  model?: {
    id: string;
    name: string;
    referenceMode: string;
  };
  usedCompositeReference?: boolean;
};

export default function Home() {
  const [userImage, setUserImage] = useState<UploadedImage | null>(null);
  const [clothingImage, setClothingImage] = useState<UploadedImage | null>(null);
  const [selectedModelId, setSelectedModelId] = useState<TryOnModelId>(DEFAULT_TRY_ON_MODEL_ID);
  const [resultImage, setResultImage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastModelName, setLastModelName] = useState('No generation yet');
  const [usedCompositeReference, setUsedCompositeReference] = useState(false);

  const selectedModel = useMemo(
    () => TRY_ON_MODELS.find((model) => model.id === selectedModelId)!,
    [selectedModelId]
  );
  const canGenerate = Boolean(userImage && clothingImage && !isProcessing);

  const handleTryOn = async () => {
    if (!userImage || !clothingImage) {
      setError('Upload both a person reference and a garment reference first.');
      return;
    }

    setIsProcessing(true);
    setError(null);
    setResultImage(null);
    setUsedCompositeReference(selectedModel.referenceMode === 'composite-reference');
    setLastModelName(selectedModel.name);

    try {
      const response = await fetch('/api/try-on', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          userImage: userImage.dataUrl,
          clothingImage: clothingImage.dataUrl,
          modelId: selectedModel.id,
        }),
      });

      const data = await response.json() as TryOnResponse;

      if (!response.ok || !data.resultImage) {
        throw new Error(data.error || 'Failed to generate the try-on.');
      }

      setResultImage(data.resultImage);
      setLastModelName(data.model?.name || selectedModel.name);
      setUsedCompositeReference(Boolean(data.usedCompositeReference));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong while generating.');
    } finally {
      setIsProcessing(false);
    }
  };

  const resetResult = () => {
    setResultImage(null);
    setError(null);
    setUsedCompositeReference(false);
    setLastModelName('No generation yet');
  };

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand-mark" href="/" aria-label="Fashion Imagine home">
          <span aria-hidden="true">FI</span>
          <strong>Fashion Imagine</strong>
        </a>
        <div className="topbar__status" aria-label="Privacy status">
          <ShieldCheck size={16} strokeWidth={1.8} />
          <span>No app-side image storage</span>
        </div>
      </header>

      <section className="studio-intro" aria-labelledby="studio-title">
        <motion.div
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35 }}
        >
          <p className="ui-eyebrow">Virtual fitting room</p>
          <h1 id="studio-title">Fit check before checkout.</h1>
          <p>
            Upload a person photo and a garment reference. The studio prepares the inputs,
            routes them through an allowlisted image model, and returns a single try-on render.
          </p>
        </motion.div>
        <div className="studio-intro__notes" aria-label="Workflow checkpoints">
          <span><CheckCircle2 size={16} strokeWidth={1.8} /> Person identity preserved</span>
          <span><CheckCircle2 size={16} strokeWidth={1.8} /> Garment details prioritized</span>
          <span><CheckCircle2 size={16} strokeWidth={1.8} /> Clear model cost mode</span>
        </div>
      </section>

      <section className="studio-grid" aria-label="Virtual try-on studio">
        <div className="reference-column">
          <ImageUploader
            id="person-upload"
            title="Person reference"
            eyebrow="Step 1"
            helper="Use a clear, well-lit full-body or half-body photo."
            kind="person"
            image={userImage}
            onImageChange={(image) => {
              setUserImage(image);
              resetResult();
            }}
          />
          <ImageUploader
            id="garment-upload"
            title="Garment reference"
            eyebrow="Step 2"
            helper="Use a clean product shot, flat lay, or model photo of the garment."
            kind="garment"
            image={clothingImage}
            onImageChange={(image) => {
              setClothingImage(image);
              resetResult();
            }}
          />
        </div>

        <ResultDisplay
          resultImage={resultImage}
          userImage={userImage}
          isProcessing={isProcessing}
          error={error}
          modelName={lastModelName}
          usedCompositeReference={usedCompositeReference}
          onReset={resetResult}
          onRetry={handleTryOn}
          canRetry={Boolean(userImage && clothingImage)}
        />

        <aside className="control-panel" aria-label="Generation controls">
          <ModelSelector
            selectedModelId={selectedModelId}
            onModelChange={(modelId) => {
              setSelectedModelId(modelId);
              resetResult();
            }}
          />

          <div className="generation-summary">
            <p className="ui-eyebrow">Selected route</p>
            <h2>{selectedModel.shortName} mode</h2>
            <p>{selectedModel.description}</p>
            <ul>
              {selectedModel.strengths.map((strength) => (
                <li key={strength}>
                  <Sparkles size={14} strokeWidth={1.8} />
                  {strength}
                </li>
              ))}
            </ul>
          </div>

          <button
            type="button"
            className="primary-button"
            onClick={handleTryOn}
            disabled={!canGenerate}
            aria-disabled={!canGenerate}
          >
            <WandSparkles size={18} strokeWidth={1.8} />
            {isProcessing ? 'Generating...' : 'Generate try-on'}
          </button>

          <div className="privacy-note">
            <LockKeyhole size={17} strokeWidth={1.8} />
            <p>
              This app does not store uploaded images. Generation sends both references to
              OpenRouter and the selected model provider.
            </p>
          </div>
        </aside>
      </section>
    </main>
  );
}
