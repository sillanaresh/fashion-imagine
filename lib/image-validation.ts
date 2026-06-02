import sharp from 'sharp';
import { imageDataUrlToBuffer } from './image-data';

export class ImageValidationError extends Error {
  status: number;

  constructor(message: string, status = 400) {
    super(message);
    this.name = 'ImageValidationError';
    this.status = status;
  }
}

export async function assertReadableImage(dataUrl: string, label: string) {
  try {
    const metadata = await sharp(imageDataUrlToBuffer(dataUrl), {
      limitInputPixels: 24_000_000,
    }).metadata();

    if (!metadata.width || !metadata.height) {
      throw new ImageValidationError(`${label} image is missing dimensions`);
    }

    return {
      width: metadata.width,
      height: metadata.height,
      format: metadata.format,
    };
  } catch (error) {
    if (error instanceof ImageValidationError) {
      throw error;
    }

    throw new ImageValidationError(`${label} image could not be decoded`);
  }
}
