import { describe, expect, it } from 'vitest';
import { createReferenceBoard } from '../lib/reference-board';

const tinyPng = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=';

describe('reference board composition', () => {
  it('creates a JPEG board for one-input image models', async () => {
    const board = await createReferenceBoard({
      userImage: tinyPng,
      clothingImage: tinyPng,
    });

    expect(board.startsWith('data:image/jpeg;base64,')).toBe(true);
    expect(board.length).toBeGreaterThan(tinyPng.length);
  });
});
