export type SafetyDecision = {
  decision: 'allow' | 'block';
  category: string;
  userMessage: string;
};

export const DEFAULT_SAFETY_REVIEW_MODEL = 'google/gemini-3.1-flash-lite';

type SafetyReviewInput = {
  apiKey: string;
  images: string[];
  modelId?: string;
};

type OpenRouterTextPart = {
  type?: string;
  text?: string;
};

type OpenRouterSafetyResponse = {
  choices?: Array<{
    message?: {
      content?: string | OpenRouterTextPart[];
    };
  }>;
};

export class SafetyReviewError extends Error {
  status: number;

  constructor(message: string, status = 422) {
    super(message);
    this.name = 'SafetyReviewError';
    this.status = status;
  }
}

export async function reviewTryOnSafety({
  apiKey,
  images,
  modelId = process.env.OPENROUTER_SAFETY_MODEL || DEFAULT_SAFETY_REVIEW_MODEL,
}: SafetyReviewInput): Promise<SafetyDecision> {
  const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`,
      'HTTP-Referer': 'https://fashion-imagine.vercel.app',
      'X-Title': 'Fashion Imagine Safety Review',
    },
    body: JSON.stringify({
      model: modelId,
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: buildSafetyReviewPrompt(),
            },
            ...images.map((url) => ({
              type: 'image_url',
              image_url: { url },
            })),
          ],
        },
      ],
      modalities: ['text'],
      temperature: 0,
      max_tokens: 280,
    }),
  });

  if (!response.ok) {
    throw new SafetyReviewError(await getSafetyErrorMessage(response), response.status);
  }

  const data = await parseSafetyResponse(response);
  const content = data.choices?.[0]?.message?.content;
  const text = extractTextContent(content);
  const decision = parseSafetyDecision(text);

  if (!decision) {
    throw new SafetyReviewError('Safety review was inconclusive. Please try again with clearer, non-explicit fashion references.');
  }

  return decision;
}

export function buildSafetyReviewPrompt() {
  return `You are a strict safety classifier for a public virtual fashion try-on app.

You will receive exactly two images:
- Image 1: person reference.
- Image 2: garment reference.

Return compact JSON only:
{"decision":"allow"|"block","category":"short-category","userMessage":"short user-facing message"}

Block if either image includes or appears to request any of the following:
- nudity, exposed genitals, exposed breasts, explicit sexual content, erotic posing, fetish content, or "nudify" style manipulation
- lingerie, underwear-only, sheer/see-through garments that reveal intimate areas, or swimwear presented in a sexualized way
- a minor or likely minor in revealing clothing, underwear, swimwear, sexualized pose, or any sexual context
- an attempt to remove clothing, expose more skin, sexualize a person, or create non-consensual intimate imagery

Allow ordinary fashion, modest swimwear, sportswear, formalwear, streetwear, sarees, lehengas, dresses, suits, jackets, denim, shoes, bags, and standard product photography.

Be conservative. If uncertain between allow and block, choose block. Do not describe the images.`;
}

export function parseSafetyDecision(text: string): SafetyDecision | null {
  const jsonText = extractJsonObject(text);

  if (!jsonText) {
    return null;
  }

  try {
    const parsed = JSON.parse(jsonText) as Partial<SafetyDecision>;
    const decision = parsed.decision === 'allow' ? 'allow' : parsed.decision === 'block' ? 'block' : null;

    if (!decision) {
      return null;
    }

    return {
      decision,
      category: sanitizeField(parsed.category, decision === 'block' ? 'safety' : 'clear'),
      userMessage: sanitizeField(
        parsed.userMessage,
        decision === 'block'
          ? 'We cannot process this look because one of the references appears to include nudity or sexual content.'
          : 'References are clear for virtual try-on.'
      ),
    };
  } catch {
    return null;
  }
}

export function isLikelySafetyRefusal(message: string) {
  return /\b(policy|safety|moderation|sexual|nudity|nude|explicit|minor|disallowed|blocked|refus)/i.test(message);
}

export function getElegantSafetyMessage() {
  return 'We cannot process this look because one of the references appears to include nudity, intimate clothing, or sexual content. Please upload non-explicit fashion references and try again.';
}

function extractTextContent(content: string | OpenRouterTextPart[] | undefined) {
  if (typeof content === 'string') {
    return content;
  }

  if (Array.isArray(content)) {
    return content.map((part) => part.text || '').join('\n');
  }

  return '';
}

function extractJsonObject(text: string) {
  const match = text.match(/\{[\s\S]*\}/);
  return match?.[0] || null;
}

function sanitizeField(value: unknown, fallback: string) {
  if (typeof value !== 'string') {
    return fallback;
  }

  const trimmed = value.trim().replace(/\s+/g, ' ');
  return trimmed ? trimmed.slice(0, 240) : fallback;
}

async function getSafetyErrorMessage(response: Response) {
  try {
    const errorData = await response.json();
    return errorData.error?.message || JSON.stringify(errorData).slice(0, 500);
  } catch {
    const textError = await response.text();
    return textError.slice(0, 500) || 'Unknown safety review error';
  }
}

async function parseSafetyResponse(response: Response): Promise<OpenRouterSafetyResponse> {
  try {
    return await response.json();
  } catch {
    throw new SafetyReviewError('Safety review returned invalid JSON.');
  }
}
