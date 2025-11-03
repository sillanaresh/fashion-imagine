import { NextRequest, NextResponse } from 'next/server';

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const GEMINI_MODEL = process.env.GEMINI_MODEL || 'google/gemini-2.5-flash-image';

export async function POST(req: NextRequest) {
  try {
    const { clothingImage, userImage } = await req.json();

    if (!clothingImage || !userImage) {
      return NextResponse.json(
        { error: 'Both clothing and user images are required' },
        { status: 400 }
      );
    }

    // Check image sizes (Vercel has a 4.5MB body size limit)
    const totalSize = (clothingImage.length + userImage.length) / 1024 / 1024; // Size in MB
    console.log(`Total image size: ${totalSize.toFixed(2)} MB`);

    if (totalSize > 4) {
      return NextResponse.json(
        { error: 'Images are too large. Please use smaller images (combined size under 4MB).' },
        { status: 413 }
      );
    }

    if (!OPENROUTER_API_KEY) {
      return NextResponse.json(
        { error: 'API key not configured' },
        { status: 500 }
      );
    }

    // Prepare the comprehensive prompt for Gemini
    const prompt = `You are an expert AI-powered virtual fashion try-on system. You will receive TWO images:

IMAGE 1: A clothing item (shirt, dress, pants, etc.) - typically from an e-commerce product photo
IMAGE 2: A person's full-body photograph

YOUR TASK: Generate a photorealistic image showing the person from IMAGE 2 wearing the clothing item from IMAGE 1.

DETAILED REQUIREMENTS:

1. CLOTHING ANALYSIS & EXTRACTION:
   - Identify the exact clothing item, its type, style, color, pattern, and design details
   - Note fabric texture, logos, prints, buttons, collars, sleeves, hemlines
   - Understand how the garment fits on the original model

2. PERSON ANALYSIS:
   - Analyze body shape, proportions, height, build
   - Identify current pose (standing, arms position, body orientation)
   - Note skin tone, lighting conditions, background
   - Detect the current clothing that needs to be replaced

3. VIRTUAL TRY-ON GENERATION:
   - Seamlessly place the clothing from IMAGE 1 onto the person from IMAGE 2
   - Maintain PERFECT alignment with the person's body shape and proportions
   - Scale the clothing appropriately to fit the person's size
   - Preserve all clothing details: colors, patterns, logos, designs
   - Create natural fabric draping and wrinkles based on body contours and pose
   - Match lighting conditions from the person's original photo
   - Add realistic shadows under arms, in folds, and at edges
   - Ensure smooth blending at clothing boundaries (neck, arms, waist, etc.)

4. PRESERVE UNCHANGED:
   - Person's face, hair, skin tone (EXACT match)
   - Background environment
   - Body pose and position
   - Hands, feet, accessories (unless they're the clothing item)
   - Overall photo quality and resolution

5. REALISM REQUIREMENTS:
   - The result must look like a natural photograph
   - No artificial edges or obvious compositing
   - Natural color harmony between clothing and environment
   - Proper perspective matching the person's pose
   - Realistic fabric physics (how the garment hangs and moves)

OUTPUT: Generate ONLY the final photorealistic image of the person wearing the new clothing. No text, no annotations, just the image.`;

    // Call OpenRouter API with Gemini model
    const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${OPENROUTER_API_KEY}`,
        'HTTP-Referer': 'https://fashion-imagine.vercel.app',
        'X-Title': 'Fashion Imagine',
      },
      body: JSON.stringify({
        model: GEMINI_MODEL,
        messages: [
          {
            role: 'user',
            content: [
              {
                type: 'text',
                text: prompt,
              },
              {
                type: 'image_url',
                image_url: {
                  url: clothingImage,
                },
              },
              {
                type: 'image_url',
                image_url: {
                  url: userImage,
                },
              },
            ],
          },
        ],
        temperature: 0.7,
        max_tokens: 4096,
      }),
    });

    if (!response.ok) {
      let errorMessage = 'Unknown error';
      try {
        const errorData = await response.json();
        console.error('OpenRouter API error:', errorData);
        errorMessage = errorData.error?.message || JSON.stringify(errorData);
      } catch (e) {
        // If response is not JSON, get it as text
        const textError = await response.text();
        console.error('OpenRouter non-JSON error:', textError);
        errorMessage = textError.substring(0, 200); // Limit error message length
      }
      return NextResponse.json(
        { error: `AI service error: ${errorMessage}` },
        { status: response.status }
      );
    }

    let data;
    try {
      data = await response.json();
    } catch (e) {
      const textResponse = await response.text();
      console.error('Failed to parse JSON response:', textResponse.substring(0, 500));
      return NextResponse.json(
        { error: 'Invalid response from AI service. Response was not JSON.' },
        { status: 500 }
      );
    }

    console.log('OpenRouter Response:', JSON.stringify(data, null, 2));

    const message = data.choices?.[0]?.message;

    if (!message) {
      return NextResponse.json(
        { error: 'No response from AI model' },
        { status: 500 }
      );
    }

    // Gemini returns images in the images array
    const images = message.images;
    const textContent = message.content;

    let resultImage = userImage; // Default fallback

    // Check if Gemini returned an image
    if (images && images.length > 0) {
      // Get the first image from the response
      const generatedImage = images[0];
      if (generatedImage.image_url && generatedImage.image_url.url) {
        resultImage = generatedImage.image_url.url;
        console.log('Successfully extracted generated image from Gemini');
      }
    } else if (textContent) {
      // Fallback: check if there's base64 in text content
      if (textContent.includes('data:image')) {
        resultImage = textContent;
      } else {
        console.log('No image generated, only text response:', textContent);
      }
    }

    return NextResponse.json({
      resultImage: resultImage,
      analysis: textContent || 'Image generated successfully',
      success: true,
    });

  } catch (error) {
    console.error('Virtual try-on error:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json(
      { error: `Failed to process virtual try-on: ${errorMessage}` },
      { status: 500 }
    );
  }
}
