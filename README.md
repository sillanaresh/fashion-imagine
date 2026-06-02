# Fashion Imagine

An AI virtual try-on studio. Upload a person reference and a garment reference, choose an allowlisted image model, and generate a single photorealistic image of the person wearing the garment.

## What It Does

1. Upload a clear person photo.
2. Upload a garment or product image.
3. Choose a generation route: value, balanced, or premium quality.
4. Generate, compare before/after, retry, or download the result.
5. If generation is blocked by safety or the GPT daily limit, the UI explains the next action without exposing raw provider errors.

## Architecture

- **Next.js App Router** renders the studio UI and exposes `/api/try-on`, `/api/usage`, and `/api/interest`.
- **Client image preparation** compresses uploads to JPEG in-browser before sending them.
- **Server validation** checks request shape, MIME type, image byte size, and decodable image metadata before any provider call.
- **Safety review** runs a cheap multimodal reviewer before expensive generation and blocks nudity, intimate clothing, sexualized edits, and non-consensual intimate-image attempts.
- **Device guardrails** use same-site cookies plus client local storage to allow one successful GPT generation per device per UTC day. The two Nano/Gemini routes are unlimited.
- **Interest signal** records one counted GPT-interest signal per device cookie for people who want more premium GPT generations.
- **Model catalog** lives in `lib/model-catalog.ts`, keeping provider routes allowlisted and explainable.
- **Reference preparation** sends the person and garment as native two-image references.
- **OpenRouter client** lives in `lib/openrouter.ts`, isolated from UI and request validation.

## Model Routes

Default route:

```env
OPENROUTER_IMAGE_MODEL=google/gemini-3.1-flash-image-preview

# Optional. Defaults to a low-cost image-input/text-output reviewer.
OPENROUTER_SAFETY_MODEL=google/gemini-3.1-flash-lite
```

Allowlisted routes:

- `google/gemini-3.1-flash-image-preview` - balanced default, native two-reference image flow.
- `google/gemini-2.5-flash-image` - lowest-cost native two-reference route in the current catalog.
- `openai/gpt-5.4-image-2` - premium OpenAI quality route currently listed by OpenRouter.

## Tech Stack

- Next.js 16 with TypeScript
- React 19
- Tailwind CSS v4 plus custom CSS tokens
- Framer Motion
- Sharp for server-side image metadata validation
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

## Guardrails

The app uses three layers:

- Request validation and image decode checks before any model call.
- A strict safety-review prompt through `OPENROUTER_SAFETY_MODEL`; inconclusive review fails closed.
- Provider refusal mapping, so moderation/policy errors become the same elegant safety state in the UI.

The daily limit only applies to the premium GPT route. The Nano/Gemini routes remain unlimited for iteration. The GPT limit is best-effort per browser/device and uses cookies plus local storage, which is appropriate for a lightweight demo but not a substitute for authenticated server-side quotas if the app becomes public at scale.

The interest counter is an in-memory server-process counter plus a device cookie. It is useful for local/demo tracing; replace `lib/interest-store.ts` with a durable database, KV store, or analytics event if you need reliable aggregate counts across deploys.

## Privacy Note

This app does not store uploaded images. During generation, both references are sent to OpenRouter and the selected model provider. Do not describe the flow as fully private unless the deployment adds provider-side retention controls, storage guarantees, and user-facing policy text.

## Project Structure

```text
app/
  api/try-on/route.ts      API orchestration
  api/usage/route.ts       Device quota/interest status
  api/interest/route.ts    One-count-per-device interest signal
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
  device-guardrails.ts     UTC day/cookie helpers for quota and interest
  fashion-facts.ts         320 global fashion notes for the loading roller
  interest-store.ts        Replaceable in-memory interest counter
  model-catalog.ts         Model allowlist and route metadata
  openrouter.ts            OpenRouter client and response extraction
  safety.ts                Multimodal safety reviewer and refusal mapping
  try-on.ts                Prompt and reference preparation
tests/                     Unit tests
e2e/                       Playwright E2E tests
```
