'use client';

import { useRef, useState } from 'react';
import { AlertCircle, ImageIcon, Shirt, UploadCloud, UserRound, X } from 'lucide-react';

export type UploadedImage = {
  dataUrl: string;
  name: string;
  sizeLabel: string;
  dimensions: string;
};

type ImageUploaderProps = {
  id: string;
  title: string;
  eyebrow: string;
  helper: string;
  image: UploadedImage | null;
  kind: 'person' | 'garment';
  onImageChange: (image: UploadedImage | null) => void;
};

export default function ImageUploader({
  id,
  title,
  eyebrow,
  helper,
  image,
  kind,
  onImageChange,
}: ImageUploaderProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const Icon = kind === 'person' ? UserRound : Shirt;

  const handleFiles = async (files: FileList | null) => {
    const file = files?.[0];
    if (!file) {
      return;
    }

    setError(null);

    try {
      const compressedImage = await compressImage(file);
      onImageChange(compressedImage);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not read this image');
    }
  };

  return (
    <section className={`upload-card ${isDragging ? 'is-dragging' : ''}`} aria-labelledby={`${id}-title`}>
      <div className="upload-card__header">
        <div className="upload-card__icon" aria-hidden="true">
          <Icon size={18} strokeWidth={1.8} />
        </div>
        <div>
          <p className="ui-eyebrow">{eyebrow}</p>
          <h2 id={`${id}-title`}>{title}</h2>
        </div>
      </div>

      <input
        ref={fileInputRef}
        id={id}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        className="sr-only"
        onChange={(event) => handleFiles(event.target.files)}
      />

      <button
        type="button"
        className="upload-dropzone"
        onClick={() => fileInputRef.current?.click()}
        onDragOver={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setIsDragging(false);
          handleFiles(event.dataTransfer.files);
        }}
      >
        {image ? (
          <span className="upload-preview">
            <img src={image.dataUrl} alt={`${title} preview`} />
          </span>
        ) : (
          <span className="upload-empty">
            <UploadCloud size={24} strokeWidth={1.8} />
            <span>Drop an image here</span>
            <small>or click to choose JPEG, PNG, or WebP</small>
          </span>
        )}
      </button>

      <div className="upload-card__footer">
        {image ? (
          <>
            <div className="upload-meta">
              <ImageIcon size={15} strokeWidth={1.8} />
              <span>{image.name}</span>
              <small>{image.dimensions} · {image.sizeLabel}</small>
            </div>
            <button
              type="button"
              className="icon-button"
              aria-label={`Remove ${title.toLowerCase()}`}
              onClick={() => {
                onImageChange(null);
                setError(null);
                if (fileInputRef.current) {
                  fileInputRef.current.value = '';
                }
              }}
            >
              <X size={16} strokeWidth={1.8} />
            </button>
          </>
        ) : (
          <p>{helper}</p>
        )}
      </div>

      {error && (
        <p className="form-error">
          <AlertCircle size={15} strokeWidth={1.8} />
          {error}
        </p>
      )}
    </section>
  );
}

async function compressImage(file: File): Promise<UploadedImage> {
  if (!file.type.startsWith('image/')) {
    throw new Error('Choose an image file');
  }

  const source = await readFileAsDataUrl(file);
  const image = await loadImage(source);
  const { width, height } = constrainDimensions(image.naturalWidth, image.naturalHeight, 1800);
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;

  const context = canvas.getContext('2d');
  if (!context) {
    throw new Error('Could not prepare this image');
  }

  context.drawImage(image, 0, 0, width, height);

  const dataUrl = canvas.toDataURL('image/jpeg', 0.84);

  return {
    dataUrl,
    name: file.name,
    sizeLabel: formatBytes(dataUrlByteLength(dataUrl)),
    dimensions: `${width} x ${height}`,
  };
}

function readFileAsDataUrl(file: File) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(new Error('Could not read this file'));
    reader.readAsDataURL(file);
  });
}

function loadImage(src: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('Could not load this image'));
    image.src = src;
  });
}

function constrainDimensions(width: number, height: number, maxSide: number) {
  if (width <= maxSide && height <= maxSide) {
    return { width, height };
  }

  const scale = maxSide / Math.max(width, height);
  return {
    width: Math.round(width * scale),
    height: Math.round(height * scale),
  };
}

function dataUrlByteLength(dataUrl: string) {
  const base64 = dataUrl.split(',')[1] || '';
  const padding = (base64.match(/=+$/)?.[0].length || 0);
  return Math.floor((base64.length * 3) / 4) - padding;
}

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) {
    return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  }

  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}
