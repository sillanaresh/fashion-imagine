'use client';

import { useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { Bell, CheckCircle2, LockKeyhole, ShieldCheck, Sparkles, WandSparkles } from 'lucide-react';
import ImageUploader, { type UploadedImage } from '@/components/ImageUploader';
import ModelSelector from '@/components/ModelSelector';
import ResultDisplay, { type ResultErrorCode } from '@/components/ResultDisplay';
import FashionMark from '@/components/FashionMark';
import {
  DEFAULT_TRY_ON_MODEL_ID,
  TRY_ON_MODELS,
  type TryOnModelId,
} from '@/lib/model-catalog';

type TryOnResponse = {
  resultImage?: string;
  error?: string;
  code?: ResultErrorCode;
  model?: {
    id: string;
    name: string;
    referenceMode: string;
  };
};

type UsageResponse = {
  generationUsedToday?: boolean;
  interestRegistered?: boolean;
  today?: string;
};

type InterestResponse = {
  selected?: boolean;
  registered?: boolean;
};

const LOCAL_GENERATION_DAY_KEY = 'fashion-imagine:generation-day';
const LOCAL_INTEREST_SELECTED_KEY = 'fashion-imagine:interest-selected';

export default function Home() {
  const [userImage, setUserImage] = useState<UploadedImage | null>(null);
  const [clothingImage, setClothingImage] = useState<UploadedImage | null>(null);
  const [selectedModelId, setSelectedModelId] = useState<TryOnModelId>(DEFAULT_TRY_ON_MODEL_ID);
  const [resultImage, setResultImage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [errorCode, setErrorCode] = useState<ResultErrorCode>(null);
  const [lastModelName, setLastModelName] = useState('No generation yet');
  const [dailyLimitUsed, setDailyLimitUsed] = useState(false);
  const [interestSelected, setInterestSelected] = useState(false);
  const [interestRegistered, setInterestRegistered] = useState(false);
  const [isInterestSaving, setIsInterestSaving] = useState(false);

  const selectedModel = useMemo(
    () => TRY_ON_MODELS.find((model) => model.id === selectedModelId)!,
    [selectedModelId]
  );
  const canGenerate = Boolean(userImage && clothingImage && !isProcessing && !dailyLimitUsed);

  useEffect(() => {
    const today = getTodayKey();
    const localLimitUsed = window.localStorage.getItem(LOCAL_GENERATION_DAY_KEY) === today;
    const localInterestSelected = window.localStorage.getItem(LOCAL_INTEREST_SELECTED_KEY) === '1';

    setDailyLimitUsed(localLimitUsed);
    setInterestSelected(localInterestSelected);

    fetch('/api/usage')
      .then((response) => response.ok ? response.json() : null)
      .then((data: UsageResponse | null) => {
        if (!data) {
          return;
        }

        const serverLimitUsed = Boolean(data.generationUsedToday);
        const serverInterestRegistered = Boolean(data.interestRegistered);
        setDailyLimitUsed(serverLimitUsed || localLimitUsed);
        setInterestRegistered(serverInterestRegistered);

        if (serverInterestRegistered) {
          setInterestSelected(true);
          window.localStorage.setItem(LOCAL_INTEREST_SELECTED_KEY, '1');
        }
      })
      .catch(() => {
        setDailyLimitUsed(localLimitUsed);
      });
  }, []);

  const handleTryOn = async () => {
    if (!userImage || !clothingImage) {
      setError('Upload both a person reference and a garment reference first.');
      setErrorCode('SYSTEM_ERROR');
      return;
    }

    if (dailyLimitUsed) {
      setError('Today’s complimentary try-on has already been used on this device. Tap Show interest if you want more generations opened up.');
      setErrorCode('DAILY_LIMIT_REACHED');
      return;
    }

    setIsProcessing(true);
    setError(null);
    setErrorCode(null);
    setResultImage(null);
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
        setErrorCode(data.code || 'SYSTEM_ERROR');
        if (data.code === 'DAILY_LIMIT_REACHED') {
          markDailyLimitUsed();
          setDailyLimitUsed(true);
        }
        throw new Error(data.error || 'Failed to generate the try-on.');
      }

      setResultImage(data.resultImage);
      setLastModelName(data.model?.name || selectedModel.name);
      markDailyLimitUsed();
      setDailyLimitUsed(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong while generating.');
    } finally {
      setIsProcessing(false);
    }
  };

  const resetResult = () => {
    setResultImage(null);
    setError(null);
    setErrorCode(null);
    setLastModelName('No generation yet');
  };

  const handleInterestToggle = async () => {
    const nextSelected = !interestSelected;
    setInterestSelected(nextSelected);
    window.localStorage.setItem(LOCAL_INTEREST_SELECTED_KEY, nextSelected ? '1' : '0');
    setIsInterestSaving(true);

    try {
      const response = await fetch('/api/interest', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ interested: nextSelected }),
      });
      const data = await response.json() as InterestResponse;

      if (response.ok) {
        setInterestRegistered(Boolean(data.registered));
      }
    } catch {
      setInterestRegistered(interestRegistered || nextSelected);
    } finally {
      setIsInterestSaving(false);
    }
  };

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand-mark" href="/" aria-label="Fashion Imagine home">
          <span className="brand-mark__symbol" aria-hidden="true">
            <FashionMark title="Fashion Imagine" />
          </span>
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
          <p className="ui-eyebrow">Virtual fitting atelier</p>
          <h1 id="studio-title">Try the look before it becomes yours.</h1>
          <p>
            Upload a person photo and a garment reference. The atelier keeps the workflow spare,
            routes both images through an allowlisted model, and returns a polished try-on render.
          </p>
        </motion.div>
        <div className="studio-intro__notes" aria-label="Workflow checkpoints">
          <span><CheckCircle2 size={16} strokeWidth={1.8} /> Preserve the wearer</span>
          <span><CheckCircle2 size={16} strokeWidth={1.8} /> Respect the garment</span>
          <span><CheckCircle2 size={16} strokeWidth={1.8} /> Choose the render house</span>
        </div>
      </section>

      <section className="studio-grid" aria-label="Virtual try-on studio">
        <div className="reference-column">
          <ImageUploader
            id="person-upload"
            title="Wearer image"
            eyebrow="Look 01"
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
            title="Garment image"
            eyebrow="Look 02"
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
          errorCode={errorCode}
          modelName={lastModelName}
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
            {isProcessing ? 'Generating...' : dailyLimitUsed ? 'Daily preview used' : 'Generate try-on'}
          </button>

          <div className={`limit-interest ${dailyLimitUsed ? 'is-active' : ''}`}>
            <div>
              <p className="ui-eyebrow">Launch controls</p>
              <h2>{dailyLimitUsed ? 'One render used today' : 'One premium render per device'}</h2>
              <p>
                {dailyLimitUsed
                  ? 'GPT Image routes are expensive, so this demo allows one successful generation per device each UTC day.'
                  : 'The first successful render is complimentary today. If you want more, leave one interest signal from this device.'}
              </p>
            </div>
            <button
              type="button"
              className={`interest-button ${interestSelected ? 'is-selected' : ''}`}
              onClick={handleInterestToggle}
              disabled={isInterestSaving}
              aria-pressed={interestSelected}
            >
              <Bell size={16} strokeWidth={1.8} />
              {interestSelected ? 'Interest noted' : 'Show interest'}
            </button>
            <small>
              {interestRegistered
                ? 'Counted once for this device. You can still toggle the button for your own reminder.'
                : 'This sends at most one counted signal per device cookie.'}
            </small>
          </div>

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

function getTodayKey() {
  return new Date().toISOString().slice(0, 10);
}

function markDailyLimitUsed() {
  window.localStorage.setItem(LOCAL_GENERATION_DAY_KEY, getTodayKey());
}
