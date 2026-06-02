import { NextRequest, NextResponse } from 'next/server';
import { z } from 'zod';
import { getDefaultTryOnModel, findTryOnModel } from '@/lib/model-catalog';
import { getImageDataUrlByteLength, isSupportedImageDataUrl } from '@/lib/image-data';
import { assertReadableImage, ImageValidationError } from '@/lib/image-validation';
import { generateTryOnImage, OpenRouterError } from '@/lib/openrouter';
import { buildVirtualTryOnPrompt, prepareTryOnReferences } from '@/lib/try-on';

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const SERVER_DEFAULT_MODEL_ID = process.env.OPENROUTER_IMAGE_MODEL;
const MAX_COMBINED_IMAGE_BYTES = 7.5 * 1024 * 1024;

const tryOnRequestSchema = z.object({
  userImage: z.string().refine(isSupportedImageDataUrl, {
    message: 'User image must be a JPEG, PNG, or WebP data URL',
  }),
  clothingImage: z.string().refine(isSupportedImageDataUrl, {
    message: 'Clothing image must be a JPEG, PNG, or WebP data URL',
  }),
  modelId: z.string().optional(),
});

export async function POST(req: NextRequest) {
  try {
    const body = await readJson(req);
    const parsed = tryOnRequestSchema.safeParse(body);

    if (!parsed.success) {
      return NextResponse.json(
        { error: parsed.error.issues[0]?.message || 'Invalid try-on request' },
        { status: 400 }
      );
    }

    const { userImage, clothingImage, modelId } = parsed.data;
    const requestedModel = modelId
      ? findTryOnModel(modelId)
      : getDefaultTryOnModel(SERVER_DEFAULT_MODEL_ID);

    if (!requestedModel) {
      return NextResponse.json(
        { error: 'Unsupported image model selected' },
        { status: 400 }
      );
    }

    const totalImageBytes = getImageDataUrlByteLength(userImage)
      + getImageDataUrlByteLength(clothingImage);

    if (totalImageBytes > MAX_COMBINED_IMAGE_BYTES) {
      return NextResponse.json(
        { error: 'Images are too large. Please use smaller images with a combined size under 7.5MB.' },
        { status: 413 }
      );
    }

    if (!OPENROUTER_API_KEY) {
      return NextResponse.json(
        { error: 'OpenRouter API key is not configured' },
        { status: 500 }
      );
    }

    await Promise.all([
      assertReadableImage(userImage, 'User'),
      assertReadableImage(clothingImage, 'Clothing'),
    ]);

    const preparedReferences = await prepareTryOnReferences({
      model: requestedModel,
      userImage,
      clothingImage,
    });
    const prompt = buildVirtualTryOnPrompt(
      requestedModel,
      preparedReferences.usedCompositeReference
    );
    const generation = await generateTryOnImage({
      apiKey: OPENROUTER_API_KEY,
      model: requestedModel,
      prompt,
      images: preparedReferences.images,
    });

    return NextResponse.json({
      resultImage: generation.resultImage,
      analysis: generation.analysis,
      model: {
        id: requestedModel.id,
        name: requestedModel.name,
        referenceMode: requestedModel.referenceMode,
      },
      usedCompositeReference: preparedReferences.usedCompositeReference,
      success: true,
    });
  } catch (error) {
    if (error instanceof OpenRouterError) {
      console.error('OpenRouter try-on generation failed:', error.message);
      return NextResponse.json(
        { error: `AI service error: ${error.message}` },
        { status: error.status }
      );
    }

    if (error instanceof ImageValidationError) {
      return NextResponse.json(
        { error: error.message },
        { status: error.status }
      );
    }

    console.error('Virtual try-on error:', error instanceof Error ? error.message : error);
    return NextResponse.json(
      { error: 'Failed to process virtual try-on' },
      { status: 500 }
    );
  }
}

async function readJson(req: NextRequest) {
  try {
    return await req.json();
  } catch {
    return null;
  }
}
