import { describe, expect, it } from 'vitest';
import {
  FASHION_FACTS,
} from '../lib/fashion-facts';
import {
  buildSafetyReviewPrompt,
  getElegantSafetyMessage,
  isLikelySafetyRefusal,
  parseSafetyDecision,
} from '../lib/safety';

describe('safety guardrails', () => {
  it('parses compact JSON decisions even when wrapped in provider text', () => {
    expect(parseSafetyDecision('Result: {"decision":"block","category":"nudity","userMessage":"Please use non-explicit fashion references."}')).toEqual({
      decision: 'block',
      category: 'nudity',
      userMessage: 'Please use non-explicit fashion references.',
    });

    expect(parseSafetyDecision('{"decision":"allow","category":"clear","userMessage":"OK"}')).toEqual({
      decision: 'allow',
      category: 'clear',
      userMessage: 'OK',
    });
  });

  it('fails parsing when the classifier output is not actionable JSON', () => {
    expect(parseSafetyDecision('looks fine')).toBeNull();
    expect(parseSafetyDecision('{"decision":"maybe"}')).toBeNull();
  });

  it('recognizes provider safety refusals and exposes a calm user message', () => {
    expect(isLikelySafetyRefusal('blocked by provider moderation policy')).toBe(true);
    expect(isLikelySafetyRefusal('temporary upstream timeout')).toBe(false);
    expect(getElegantSafetyMessage()).toContain('non-explicit fashion references');
  });

  it('instructs the reviewer to block intimate and non-consensual edits', () => {
    const prompt = buildSafetyReviewPrompt();
    expect(prompt).toContain('nudity');
    expect(prompt).toContain('non-consensual intimate imagery');
    expect(prompt).toContain('"decision":"allow"|"block"');
  });

  it('ships a long global fashion fact set for loading states', () => {
    expect(FASHION_FACTS).toHaveLength(320);
    expect(new Set(FASHION_FACTS).size).toBe(320);
  });
});
