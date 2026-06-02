import type { TryOnModel } from './model-catalog';

type OpenRouterImage = {
  image_url?: {
    url?: string;
  };
  imageUrl?: {
    url?: string;
  };
};

type OpenRouterMessage = {
  content?: string;
  images?: OpenRouterImage[];
};

type OpenRouterChoice = {
  message?: OpenRouterMessage;
};

type OpenRouterResponse = {
  choices?: OpenRouterChoice[];
};

type GenerateTryOnImageInput = {
  apiKey: string;
  model: TryOnModel;
  prompt: string;
  images: string[];
};

export class OpenRouterError extends Error {
  status: number;

  constructor(message: string, status = 500) {
    super(message);
    this.name = 'OpenRouterError';
    this.status = status;
  }
}

export async function generateTryOnImage({
  apiKey,
  model,
  prompt,
  images,
}: GenerateTryOnImageInput) {
  const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`,
      'HTTP-Referer': 'https://fashion-imagine.vercel.app',
      'X-Title': 'Fashion Imagine',
    },
    body: JSON.stringify({
      model: model.id,
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: prompt,
            },
            ...images.map((url) => ({
              type: 'image_url',
              image_url: { url },
            })),
          ],
        },
      ],
      modalities: ['image', 'text'],
      max_tokens: 4096,
    }),
  });

  if (!response.ok) {
    throw new OpenRouterError(await getOpenRouterErrorMessage(response), response.status);
  }

  const data = await parseOpenRouterResponse(response);
  const message = data.choices?.[0]?.message;

  if (!message) {
    throw new OpenRouterError('No response from AI model');
  }

  const resultImage = extractGeneratedImage(message);

  if (!resultImage) {
    throw new OpenRouterError('No generated image returned by AI model');
  }

  return {
    resultImage,
    analysis: message.content || 'Image generated successfully',
  };
}

export function extractGeneratedImage(message: OpenRouterMessage) {
  const generatedImage = message.images?.find((image) => (
    image.image_url?.url || image.imageUrl?.url
  ));

  if (generatedImage) {
    return generatedImage.image_url?.url || generatedImage.imageUrl?.url || null;
  }

  if (message.content?.includes('data:image')) {
    return message.content;
  }

  return null;
}

async function getOpenRouterErrorMessage(response: Response) {
  try {
    const errorData = await response.json();
    return errorData.error?.message || JSON.stringify(errorData).slice(0, 500);
  } catch {
    const textError = await response.text();
    return textError.slice(0, 500) || 'Unknown AI service error';
  }
}

async function parseOpenRouterResponse(response: Response): Promise<OpenRouterResponse> {
  try {
    return await response.json();
  } catch {
    throw new OpenRouterError('Invalid response from AI service. Response was not JSON.');
  }
}
