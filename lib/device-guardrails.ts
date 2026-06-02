export const GENERATION_LIMIT_COOKIE = 'fi_generation_day';
export const INTEREST_SIGNAL_COOKIE = 'fi_interest_signal';

export function getUtcDayKey(date = new Date()) {
  return date.toISOString().slice(0, 10);
}

export function secondsUntilNextUtcDay(date = new Date()) {
  const nextDay = new Date(Date.UTC(
    date.getUTCFullYear(),
    date.getUTCMonth(),
    date.getUTCDate() + 1,
    0,
    0,
    0,
    0
  ));

  return Math.max(60, Math.ceil((nextDay.getTime() - date.getTime()) / 1000));
}

export function getCookieOptions(maxAge: number) {
  return {
    httpOnly: true,
    sameSite: 'lax' as const,
    secure: process.env.NODE_ENV === 'production',
    path: '/',
    maxAge,
  };
}
