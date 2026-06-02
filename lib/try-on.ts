import type { TryOnModel } from './model-catalog';
import { createReferenceBoard } from './reference-board';

type PrepareTryOnReferencesInput = {
  model: TryOnModel;
  userImage: string;
  clothingImage: string;
};

export async function prepareTryOnReferences({
  model,
  userImage,
  clothingImage,
}: PrepareTryOnReferencesInput) {
  if (model.referenceMode === 'composite-reference') {
    return {
      images: [
        await createReferenceBoard({ userImage, clothingImage }),
      ],
      usedCompositeReference: true,
    };
  }

  return {
    images: [userImage, clothingImage],
    usedCompositeReference: false,
  };
}

export function buildVirtualTryOnPrompt(model: TryOnModel, usedCompositeReference: boolean) {
  const referenceInstructions = usedCompositeReference
    ? `You receive ONE reference board image. The left panel is labeled PERSON REFERENCE and the right panel is labeled GARMENT REFERENCE. Use those labels only to interpret the references. Do not include the board, labels, panel frames, or any text in the output.`
    : `You receive TWO images. IMAGE 1 is the person's photograph. IMAGE 2 is the clothing/product reference.`;

  return `You are an expert virtual fashion try-on image editor.

${referenceInstructions}

Goal:
Generate one photorealistic image of the person wearing the garment from the clothing reference.

Rules:
- Preserve the person's face, hair, skin tone, body shape, pose, camera angle, background, hands, feet, and accessories unless the target garment naturally covers them.
- Replace only the relevant clothing region with the referenced garment.
- Preserve garment identity: color, cut, fabric texture, print, logo placement, buttons, collar, hem, sleeve length, drape, and silhouette.
- If the clothing reference shows a mannequin, hanger, model, flat lay, or product background, ignore those and extract only the garment.
- Fit the garment to the person's proportions without changing their body shape.
- Match the original photo lighting, shadows, lens perspective, and grain.
- Avoid artificial seams, pasted edges, distorted hands, warped logos, extra limbs, duplicate garments, floating fabric, and text artifacts.
- Keep the output in the same overall framing as the person's photo when possible.

Model-specific note:
${model.name} is being used in ${model.referenceMode === 'native-two-image' ? 'native two-reference mode' : 'composite-reference mode'}.

Output:
Return only the final try-on image. No captions, no annotations, no comparison grid.`;
}
