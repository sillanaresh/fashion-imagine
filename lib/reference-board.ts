import sharp from 'sharp';
import { imageDataUrlToBuffer } from './image-data';

const BOARD_WIDTH = 1800;
const BOARD_HEIGHT = 1200;
const PANEL_WIDTH = 820;
const PANEL_HEIGHT = 980;
const PANEL_TOP = 132;
const PERSON_LEFT = 60;
const GARMENT_LEFT = 920;

type ReferenceBoardInput = {
  userImage: string;
  clothingImage: string;
};

export async function createReferenceBoard({ userImage, clothingImage }: ReferenceBoardInput) {
  const [person, garment] = await Promise.all([
    preparePanelImage(userImage),
    preparePanelImage(clothingImage),
  ]);

  const chrome = Buffer.from(`
    <svg width="${BOARD_WIDTH}" height="${BOARD_HEIGHT}" viewBox="0 0 ${BOARD_WIDTH} ${BOARD_HEIGHT}" xmlns="http://www.w3.org/2000/svg">
      <rect width="100%" height="100%" fill="#f5f6f0"/>
      ${panel(PERSON_LEFT, PANEL_TOP, 'PERSON REFERENCE')}
      ${panel(GARMENT_LEFT, PANEL_TOP, 'GARMENT REFERENCE')}
      <text x="${BOARD_WIDTH / 2}" y="70" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="28" font-weight="700" fill="#20231f">Virtual try-on reference board</text>
      <text x="${BOARD_WIDTH / 2}" y="110" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="18" fill="#5a5f55">Use the left person and the right garment. Do not render this board, labels, or frame.</text>
    </svg>
  `);

  const board = await sharp({
    create: {
      width: BOARD_WIDTH,
      height: BOARD_HEIGHT,
      channels: 3,
      background: '#f5f6f0',
    },
  })
    .composite([
      { input: chrome, left: 0, top: 0 },
      { input: person, left: PERSON_LEFT + 40, top: PANEL_TOP + 92 },
      { input: garment, left: GARMENT_LEFT + 40, top: PANEL_TOP + 92 },
    ])
    .jpeg({ quality: 88, mozjpeg: true })
    .toBuffer();

  return `data:image/jpeg;base64,${board.toString('base64')}`;
}

async function preparePanelImage(dataUrl: string) {
  return sharp(imageDataUrlToBuffer(dataUrl))
    .rotate()
    .resize({
      width: PANEL_WIDTH - 80,
      height: PANEL_HEIGHT - 132,
      fit: 'inside',
      withoutEnlargement: true,
    })
    .flatten({ background: '#ffffff' })
    .jpeg({ quality: 88, mozjpeg: true })
    .toBuffer();
}

function panel(left: number, top: number, label: string) {
  return `
    <rect x="${left}" y="${top}" width="${PANEL_WIDTH}" height="${PANEL_HEIGHT}" rx="18" fill="#ffffff" stroke="#c9cec0" stroke-width="2"/>
    <rect x="${left + 22}" y="${top + 22}" width="${PANEL_WIDTH - 44}" height="48" rx="8" fill="#20231f"/>
    <text x="${left + 44}" y="${top + 54}" font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="700" fill="#ffffff">${label}</text>
  `;
}
