# Fashion Imagine

An AI virtual try-on studio. Upload a person reference and a garment reference, choose an allowlisted image model, and generate a single photorealistic image of the person wearing the garment.

## What It Does

1. Upload a clear person photo.
2. Upload a garment or product image.
3. Choose a generation route: value, balanced, quality, or experimental Recraft.
4. Generate, compare before/after, retry, or download the result.

## Architecture

- **Next.js App Router** renders the studio UI and exposes `/api/try-on`.
- **Client image preparation** compresses uploads to JPEG in-browser before sending them.
- **Server validation** checks request shape, MIME type, image byte size, and decodable image metadata before any provider call.
- **Model catalog** lives in `lib/model-catalog.ts`, keeping provider routes allowlisted and explainable.
- **Reference preparation** supports native two-image models and one-image models. One-image routes receive a generated reference board composed with Sharp.
- **OpenRouter client** lives in `lib/openrouter.ts`, isolated from UI and request validation.

## Model Routes

Default route:

```env
OPENROUTER_IMAGE_MODEL=google/gemini-3.1-flash-image-preview
```

Allowlisted routes:

- `google/gemini-3.1-flash-image-preview` - balanced default, native two-reference image flow.
- `google/gemini-2.5-flash-image` - lowest-cost native two-reference route in the current catalog.
- `openai/gpt-5.4-image-2` - premium OpenAI quality route currently listed by OpenRouter.
- `recraft/recraft-v4.1-utility-pro` - experimental route. OpenRouter notes Recraft supports one input image, so the app sends a composed reference board.

## Tech Stack

- Next.js 16 with TypeScript
- React 19
- Tailwind CSS v4 plus custom CSS tokens
- Framer Motion
- Sharp for server-side reference-board composition
- Zod for API request validation
- Vitest and Playwright for tests
- OpenRouter for image model access

## Getting Started

```bash
npm install
```

Create `.env.local`:

```env
OPENROUTER_API_KEY=your_api_key_here

# Optional. Defaults to the balanced route.
OPENROUTER_IMAGE_MODEL=google/gemini-3.1-flash-image-preview
```

Run locally:

```bash
npm run dev
```

Open `http://localhost:3000`.

## Scripts

```bash
npm test       # unit tests
npm run test:e2e
npm run build
npm audit
```

The E2E suite mocks `/api/try-on`, so it does not spend model credits or require `OPENROUTER_API_KEY`.

## Privacy Note

This app does not store uploaded images. During generation, both references are sent to OpenRouter and the selected model provider. Do not describe the flow as fully private unless the deployment adds provider-side retention controls, storage guarantees, and user-facing policy text.

## Project Structure

```text
app/
  api/try-on/route.ts      API orchestration
  globals.css              Design tokens and responsive studio UI
  layout.tsx               Metadata and root layout
  page.tsx                 Studio workflow
components/
  ImageUploader.tsx        Drag/drop upload with compression
  ModelSelector.tsx        Allowlisted generation routes
  ResultDisplay.tsx        Empty, loading, error, result, compare states
lib/
  image-data.ts            Data URL parsing and byte accounting
  image-validation.ts      Server-side image decode validation
  model-catalog.ts         Model allowlist and route metadata
  openrouter.ts            OpenRouter client and response extraction
  reference-board.ts       Sharp-composed reference board for one-image models
  try-on.ts                Prompt and reference preparation
tests/                     Unit tests
e2e/                       Playwright E2E tests
```
