export const SUPPORTED_IMAGE_MIME_TYPES = ['image/jpeg', 'image/png', 'image/webp'] as const;

export type SupportedImageMimeType = typeof SUPPORTED_IMAGE_MIME_TYPES[number];

export type ImageDataUrlInfo = {
  mimeType: SupportedImageMimeType;
  base64: string;
  byteLength: number;
};

const IMAGE_DATA_URL_PATTERN = /^data:(image\/(?:jpe?g|png|webp));base64,([a-z0-9+/=\s]+)$/i;

export function parseImageDataUrl(value: unknown): ImageDataUrlInfo | null {
  if (typeof value !== 'string') {
    return null;
  }

  const match = IMAGE_DATA_URL_PATTERN.exec(value);
  if (!match) {
    return null;
  }

  const mimeType = normalizeMimeType(match[1]);
  if (!mimeType) {
    return null;
  }

  const base64 = match[2].replace(/\s/g, '');
  let byteLength = 0;

  try {
    byteLength = Buffer.from(base64, 'base64').byteLength;
  } catch {
    return null;
  }

  return {
    mimeType,
    base64,
    byteLength,
  };
}

export function isSupportedImageDataUrl(value: unknown) {
  return parseImageDataUrl(value) !== null;
}

export function imageDataUrlToBuffer(value: string) {
  const parsed = parseImageDataUrl(value);

  if (!parsed) {
    throw new Error('Unsupported image data URL');
  }

  return Buffer.from(parsed.base64, 'base64');
}

export function getImageDataUrlByteLength(value: string) {
  return parseImageDataUrl(value)?.byteLength || 0;
}

function normalizeMimeType(mimeType: string): SupportedImageMimeType | null {
  const normalized = mimeType.toLowerCase() === 'image/jpg'
    ? 'image/jpeg'
    : mimeType.toLowerCase();

  if (SUPPORTED_IMAGE_MIME_TYPES.includes(normalized as SupportedImageMimeType)) {
    return normalized as SupportedImageMimeType;
  }

  return null;
}
