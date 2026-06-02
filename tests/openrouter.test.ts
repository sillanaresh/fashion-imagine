import { describe, expect, it } from 'vitest';
import { extractGeneratedImage } from '../lib/openrouter';

describe('OpenRouter response extraction', () => {
  it('extracts images from snake_case image_url responses', () => {
    expect(extractGeneratedImage({
      images: [{ image_url: { url: 'data:image/png;base64,abc' } }],
    })).toBe('data:image/png;base64,abc');
  });

  it('extracts images from camelCase imageUrl responses', () => {
    expect(extractGeneratedImage({
      images: [{ imageUrl: { url: 'https://example.com/result.png' } }],
    })).toBe('https://example.com/result.png');
  });

  it('falls back to data image content and rejects plain text', () => {
    expect(extractGeneratedImage({ content: 'data:image/jpeg;base64,xyz' })).toBe('data:image/jpeg;base64,xyz');
    expect(extractGeneratedImage({ content: 'No image generated' })).toBeNull();
  });
});
