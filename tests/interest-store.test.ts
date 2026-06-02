import { describe, expect, it } from 'vitest';
import { getInterestSignalCount, recordInterestSignal } from '../lib/interest-store';

describe('interest store', () => {
  it('counts interest signals in the current server process', () => {
    const before = getInterestSignalCount();
    expect(recordInterestSignal()).toBe(before + 1);
    expect(getInterestSignalCount()).toBe(before + 1);
  });
});
