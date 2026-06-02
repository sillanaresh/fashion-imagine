import { NextRequest, NextResponse } from 'next/server';
import {
  GENERATION_LIMIT_COOKIE,
  INTEREST_SIGNAL_COOKIE,
  getUtcDayKey,
} from '@/lib/device-guardrails';

export async function GET(req: NextRequest) {
  const today = getUtcDayKey();

  return NextResponse.json({
    generationUsedToday: req.cookies.get(GENERATION_LIMIT_COOKIE)?.value === today,
    interestRegistered: req.cookies.get(INTEREST_SIGNAL_COOKIE)?.value === '1',
    today,
  });
}
