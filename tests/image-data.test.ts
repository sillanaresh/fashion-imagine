import { describe, expect, it } from 'vitest';
import {
  getImageDataUrlByteLength,
  isSupportedImageDataUrl,
  parseImageDataUrl,
} from '../lib/image-data';

const tinyPng = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=';

describe('image data URL helpers', () => {
  it('accepts supported image data URLs and reports bytes', () => {
    const parsed = parseImageDataUrl(tinyPng);

    expect(parsed?.mimeType).toBe('image/png');
    expect(parsed?.byteLength).toBeGreaterThan(0);
    expect(getImageDataUrlByteLength(tinyPng)).toBe(parsed?.byteLength);
  });

  it('rejects unsupported MIME types and ordinary strings', () => {
    expect(isSupportedImageDataUrl('hello')).toBe(false);
    expect(isSupportedImageDataUrl('data:text/plain;base64,aGVsbG8=')).toBe(false);
  });

  it('normalizes image/jpg to image/jpeg', () => {
    const parsed = parseImageDataUrl('data:image/jpg;base64,aGVsbG8=');
    expect(parsed?.mimeType).toBe('image/jpeg');
  });
});
