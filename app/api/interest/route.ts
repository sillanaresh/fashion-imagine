import { NextRequest, NextResponse } from 'next/server';
import { z } from 'zod';
import {
  INTEREST_SIGNAL_COOKIE,
  getCookieOptions,
} from '@/lib/device-guardrails';
import { getInterestSignalCount, recordInterestSignal } from '@/lib/interest-store';

const interestSchema = z.object({
  interested: z.boolean(),
});

export async function POST(req: NextRequest) {
  const parsed = interestSchema.safeParse(await readJson(req));

  if (!parsed.success) {
    return NextResponse.json(
      { error: 'Invalid interest signal' },
      { status: 400 }
    );
  }

  const alreadyRegistered = req.cookies.get(INTEREST_SIGNAL_COOKIE)?.value === '1';
  const countedThisRequest = parsed.data.interested && !alreadyRegistered;
  const interestCount = countedThisRequest
    ? recordInterestSignal()
    : getInterestSignalCount();
  const response = NextResponse.json({
    selected: parsed.data.interested,
    registered: parsed.data.interested || alreadyRegistered,
    countedThisRequest,
    interestCount,
  });

  if (countedThisRequest) {
    response.cookies.set(INTEREST_SIGNAL_COOKIE, '1', getCookieOptions(60 * 60 * 24 * 365));
    console.info('Fashion Imagine interest signal registered', { interestCount });
  }

  return response;
}

async function readJson(req: NextRequest) {
  try {
    return await req.json();
  } catch {
    return null;
  }
}
