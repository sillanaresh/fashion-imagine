'use client';

import { useEffect, useState } from 'react';
import { Columns2, Download, RefreshCcw, RotateCcw, Sparkles } from 'lucide-react';
import type { UploadedImage } from './ImageUploader';

type ResultDisplayProps = {
  resultImage: string | null;
  userImage: UploadedImage | null;
  isProcessing: boolean;
  error: string | null;
  modelName: string;
  onReset: () => void;
  onRetry: () => void;
  canRetry: boolean;
};

export default function ResultDisplay({
  resultImage,
  userImage,
  isProcessing,
  error,
  modelName,
  onReset,
  onRetry,
  canRetry,
}: ResultDisplayProps) {
  const [view, setView] = useState<'result' | 'compare'>('result');
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const canCompare = Boolean(resultImage && userImage);
  const loadingStage = getLoadingStage(elapsedSeconds);

  useEffect(() => {
    if (!isProcessing) {
      setElapsedSeconds(0);
      return;
    }

    const startedAt = Date.now();
    const timer = window.setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - startedAt) / 1000));
    }, 1000);

    return () => window.clearInterval(timer);
  }, [isProcessing]);

  return (
    <section className="result-stage" aria-label="Generated try-on result">
      <div className="result-stage__header">
        <div>
          <p className="ui-eyebrow">Output</p>
          <h2>Generated fitting</h2>
        </div>
        <div className="result-tabs" role="tablist" aria-label="Result view">
          <button
            type="button"
            className={view === 'result' ? 'is-active' : ''}
            onClick={() => setView('result')}
            role="tab"
            aria-selected={view === 'result'}
          >
            <Sparkles size={15} strokeWidth={1.8} />
            Result
          </button>
          <button
            type="button"
            className={view === 'compare' ? 'is-active' : ''}
            onClick={() => setView('compare')}
            disabled={!canCompare}
            role="tab"
            aria-selected={view === 'compare'}
          >
            <Columns2 size={15} strokeWidth={1.8} />
            Compare
          </button>
        </div>
      </div>

      <div className="result-canvas">
        {isProcessing && (
          <div className="result-empty" aria-live="polite">
            <span className="spinner" aria-hidden="true" />
            <strong>{loadingStage.title}</strong>
            <p>{loadingStage.copy}</p>
            <span className="generation-timer">{elapsedSeconds}s elapsed</span>
          </div>
        )}

        {!isProcessing && error && (
          <div className="result-empty is-error" aria-live="polite">
            <strong>Generation failed</strong>
            <p>{error}</p>
          </div>
        )}

        {!isProcessing && !error && !resultImage && (
          <div className="result-empty">
            <Sparkles size={28} strokeWidth={1.6} />
            <strong>Ready for a fitting</strong>
            <p>Upload both references, choose a model, then generate the final try-on.</p>
          </div>
        )}

        {!isProcessing && !error && resultImage && view === 'result' && (
          <img src={resultImage} alt="Generated virtual try-on" className="result-image" />
        )}

        {!isProcessing && !error && resultImage && view === 'compare' && userImage && (
          <div className="compare-grid">
            <figure>
              <img src={userImage.dataUrl} alt="Original person reference" />
              <figcaption>Before</figcaption>
            </figure>
            <figure>
              <img src={resultImage} alt="Generated try-on comparison" />
              <figcaption>After</figcaption>
            </figure>
          </div>
        )}
      </div>

      <div className="result-stage__footer">
        <div className="result-model">
          <span>{modelName}</span>
          <small>Native two-reference</small>
        </div>
        <div className="result-actions">
          {resultImage && (
            <a className="secondary-button" href={resultImage} download="fashion-imagine-result.jpg">
              <Download size={16} strokeWidth={1.8} />
              Download
            </a>
          )}
          <button type="button" className="secondary-button" onClick={onReset} disabled={!resultImage && !error}>
            <RotateCcw size={16} strokeWidth={1.8} />
            Clear
          </button>
          <button type="button" className="secondary-button" onClick={onRetry} disabled={!canRetry || isProcessing}>
            <RefreshCcw size={16} strokeWidth={1.8} />
            Retry
          </button>
        </div>
      </div>
    </section>
  );
}

function getLoadingStage(elapsedSeconds: number) {
  if (elapsedSeconds >= 45) {
    return {
      title: 'Still rendering',
      copy: 'Premium image models can take a minute. The request is still active unless an error appears.',
    };
  }

  if (elapsedSeconds >= 20) {
    return {
      title: 'Refining fabric and fit',
      copy: 'This is normal for high-fidelity try-ons. The model is preserving identity and garment details.',
    };
  }

  return {
    title: 'Generating the try-on',
    copy: 'The model is aligning the garment to the person reference.',
  };
}
