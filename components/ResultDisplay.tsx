'use client';

import { useEffect, useState } from 'react';
import { AlertTriangle, Columns2, Download, RefreshCcw, RotateCcw, ShieldCheck, Sparkles } from 'lucide-react';
import { FASHION_FACTS } from '@/lib/fashion-facts';
import type { UploadedImage } from './ImageUploader';

export type ResultErrorCode =
  | 'SAFETY_BLOCKED'
  | 'SAFETY_REVIEW_UNAVAILABLE'
  | 'DAILY_LIMIT_REACHED'
  | 'AI_SERVICE_ERROR'
  | 'SYSTEM_ERROR'
  | null;

type ResultDisplayProps = {
  resultImage: string | null;
  userImage: UploadedImage | null;
  isProcessing: boolean;
  error: string | null;
  errorCode: ResultErrorCode;
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
  errorCode,
  modelName,
  onReset,
  onRetry,
  canRetry,
}: ResultDisplayProps) {
  const [view, setView] = useState<'result' | 'compare'>('result');
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [factIndex, setFactIndex] = useState(0);
  const canCompare = Boolean(resultImage && userImage);
  const loadingStage = getLoadingStage(elapsedSeconds);
  const errorPresentation = getErrorPresentation(errorCode);

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

  useEffect(() => {
    if (!isProcessing) {
      setFactIndex(0);
      return;
    }

    const factTimer = window.setInterval(() => {
      setFactIndex((current) => (current + 1) % FASHION_FACTS.length);
    }, 4200);

    return () => window.clearInterval(factTimer);
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
          <div className="result-loading" aria-live="polite">
            <div className="result-loading__status">
              <span className="spinner" aria-hidden="true" />
              <div>
                <strong>{loadingStage.title}</strong>
                <p>{loadingStage.copy}</p>
              </div>
              <span className="generation-timer">{elapsedSeconds}s</span>
            </div>

            <div className="fashion-fact-roller" aria-label="Fashion fact while rendering">
              <p className="ui-eyebrow">While the atelier works</p>
              <div className="fashion-fact-roller__window">
                <span>{FASHION_FACTS[factIndex]}</span>
              </div>
              <small>{factIndex + 1} / {FASHION_FACTS.length} global fashion notes</small>
            </div>
          </div>
        )}

        {!isProcessing && error && (
          <div className={`result-empty is-error ${errorPresentation.className}`} aria-live="polite">
            {errorPresentation.icon === 'shield' ? (
              <ShieldCheck size={30} strokeWidth={1.6} />
            ) : (
              <AlertTriangle size={30} strokeWidth={1.6} />
            )}
            <strong>{errorPresentation.title}</strong>
            <p>{error}</p>
            <small>{errorPresentation.hint}</small>
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
      copy: 'Our digital experts are still pinning fabric in place. Premium models take their sweet, expensive time.',
    };
  }

  if (elapsedSeconds >= 20) {
    return {
      title: 'Refining fabric and fit',
      copy: 'This is normal for high-fidelity try-ons. We are guarding the wearer and preserving the garment details.',
    };
  }

  return {
    title: 'Atelier in progress',
    copy: 'Our digital experts are processing the look. It may take a little while; the fabric committee is very serious.',
  };
}

function getErrorPresentation(errorCode: ResultErrorCode) {
  if (errorCode === 'SAFETY_BLOCKED' || errorCode === 'SAFETY_REVIEW_UNAVAILABLE') {
    return {
      className: 'is-safety',
      icon: 'shield',
      title: 'We cannot process this look',
      hint: 'Use clear, non-explicit fashion references. The app blocks intimate or sexualized edits before generation.',
    };
  }

  if (errorCode === 'DAILY_LIMIT_REACHED') {
    return {
      className: 'is-limit',
      icon: 'alert',
      title: 'Daily preview used',
      hint: 'One complimentary generation is available per device each UTC day.',
    };
  }

  return {
    className: 'is-system',
    icon: 'alert',
    title: 'Generation failed',
    hint: 'You can retry with clearer references or another model.',
  };
}
