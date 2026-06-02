import { NextRequest, NextResponse } from 'next/server';
import {
  GENERATION_LIMIT_COOKIE,
  INTEREST_SIGNAL_COOKIE,
  getUtcDayKey,
} from '@/lib/device-guardrails';
import { getInterestSignalCount } from '@/lib/interest-store';

export async function GET(req: NextRequest) {
  const today = getUtcDayKey();

  return NextResponse.json({
    gptGenerationUsedToday: req.cookies.get(GENERATION_LIMIT_COOKIE)?.value === today,
    generationUsedToday: req.cookies.get(GENERATION_LIMIT_COOKIE)?.value === today,
    interestRegistered: req.cookies.get(INTEREST_SIGNAL_COOKIE)?.value === '1',
    interestCount: getInterestSignalCount(),
    today,
  });
}
