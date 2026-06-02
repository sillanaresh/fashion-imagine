import { describe, expect, it } from 'vitest';
import { assertReadableImage, ImageValidationError } from '../lib/image-validation';

const tinyPng = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=';

describe('server image validation', () => {
  it('accepts readable image bytes', async () => {
    const metadata = await assertReadableImage(tinyPng, 'User');
    expect(metadata.width).toBeGreaterThan(0);
    expect(metadata.height).toBeGreaterThan(0);
  });

  it('rejects base64 payloads that are not image bytes', async () => {
    await expect(assertReadableImage('data:image/jpeg;base64,aGVsbG8=', 'User'))
      .rejects
      .toBeInstanceOf(ImageValidationError);
  });
});
