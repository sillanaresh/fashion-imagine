import { describe, expect, it } from 'vitest';
import { getUtcDayKey, secondsUntilNextUtcDay } from '../lib/device-guardrails';

describe('device guardrail helpers', () => {
  it('uses UTC day keys for daily generation limits', () => {
    expect(getUtcDayKey(new Date('2026-06-02T23:59:59.000Z'))).toBe('2026-06-02');
    expect(getUtcDayKey(new Date('2026-06-03T00:00:00.000Z'))).toBe('2026-06-03');
  });

  it('computes cookie lifetime until the next UTC day', () => {
    expect(secondsUntilNextUtcDay(new Date('2026-06-02T23:59:30.000Z'))).toBe(60);
    expect(secondsUntilNextUtcDay(new Date('2026-06-02T12:00:00.000Z'))).toBe(43200);
  });
});
