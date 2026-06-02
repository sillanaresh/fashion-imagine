import { expect, test, type Page, type TestInfo } from '@playwright/test';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';

const fixtureBase64 = 'iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAIAAAAC64paAAAAI0lEQVR4nGP8z0A+YKJA76jmUc2jmkc1j2oe1TxqkwEA8rQCHfF9QJUAAAAASUVORK5CYII=';
const resultDataUrl = `data:image/png;base64,${fixtureBase64}`;

test('generates a try-on result from two uploaded references', async ({ page }, testInfo) => {
  await mockTryOn(page, 200, {
    resultImage: resultDataUrl,
    model: {
      id: 'google/gemini-3.1-flash-image-preview',
      name: 'Mock Balanced Model',
      referenceMode: 'native-two-image',
    },
    success: true,
  });

  await page.goto('/');
  await expect(page.getByRole('button', { name: 'Generate try-on' })).toBeDisabled();

  await uploadFixturePair(page, testInfo);
  await expect(page.getByRole('button', { name: 'Generate try-on' })).toBeEnabled();

  await page.getByRole('button', { name: 'Generate try-on' }).click();
  await expect(page.getByAltText('Generated virtual try-on')).toBeVisible();
  await expect(page.getByText('Mock Balanced Model')).toBeVisible();
  await expect(page.getByRole('link', { name: 'Download' })).toHaveAttribute('download', 'fashion-imagine-result.jpg');

  await page.getByRole('tab', { name: 'Compare' }).click();
  await expect(page.getByText('Before', { exact: true })).toBeVisible();
  await expect(page.getByText('After', { exact: true })).toBeVisible();
});

test('shows an actionable error when the API fails', async ({ page }, testInfo) => {
  await mockTryOn(page, 500, {
    error: 'Mock provider failure',
  });

  await page.goto('/');
  await uploadFixturePair(page, testInfo);
  await page.getByRole('button', { name: 'Generate try-on' }).click();

  await expect(page.getByText('Generation failed')).toBeVisible();
  await expect(page.getByText('Mock provider failure')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Retry' })).toBeEnabled();
});

test('shows a safety-specific refusal when references are blocked', async ({ page }, testInfo) => {
  await mockTryOn(page, 422, {
    code: 'SAFETY_BLOCKED',
    error: 'We cannot process this look because one of the references appears to include nudity or sexual content.',
  });

  await page.goto('/');
  await uploadFixturePair(page, testInfo);
  await page.getByRole('button', { name: 'Generate try-on' }).click();

  await expect(page.getByText('We cannot process this look', { exact: true })).toBeVisible();
  await expect(page.getByText('non-explicit fashion references')).toBeVisible();
});

test('moves long-running generation feedback to the top and shows rolling facts', async ({ page }, testInfo) => {
  await mockTryOn(page, 200, {
    resultImage: resultDataUrl,
    model: {
      id: 'google/gemini-3.1-flash-image-preview',
      name: 'Mock Balanced Model',
      referenceMode: 'native-two-image',
    },
    success: true,
  }, 900);

  await page.goto('/');
  await uploadFixturePair(page, testInfo);
  await page.getByRole('button', { name: 'Generate try-on' }).click();

  await expect(page.getByText('Atelier in progress')).toBeVisible();
  await expect(page.getByText('While the atelier works')).toBeVisible();
  await expect(page.getByText('global fashion notes')).toBeHidden();
});

test('keeps Nano routes unlimited and gates only the GPT route after the daily limit', async ({ page }, testInfo) => {
  let interestSignals = 0;

  await page.route('**/api/usage', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        gptGenerationUsedToday: true,
        interestRegistered: false,
        today: '2026-06-02',
      }),
    });
  });

  await page.route('**/api/interest', async (route) => {
    const body = route.request().postDataJSON() as { interested: boolean };

    if (body.interested) {
      interestSignals += 1;
    }

    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        selected: body.interested,
        registered: interestSignals > 0,
        countedThisRequest: body.interested && interestSignals === 1,
      }),
    });
  });

  await page.goto('/');
  await uploadFixturePair(page, testInfo);

  await expect(page.getByRole('button', { name: 'Generate try-on' })).toBeEnabled();
  await expect(page.getByText('Nano routes are unlimited today')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Show interest' })).toBeHidden();

  await page.locator('input[value="openai/gpt-5.4-image-2"]').check({ force: true });

  await expect(page.getByRole('button', { name: 'GPT preview used' })).toBeDisabled();
  await expect(page.getByText('Free GPT used today')).toBeVisible();

  await page.getByRole('button', { name: 'Show interest' }).click();
  await expect(page.getByRole('button', { name: 'Interest noted' })).toBeVisible();
  await expect(page.getByText('Counted once for this device')).toBeVisible();

  await page.getByRole('button', { name: 'Interest noted' }).click();
  await expect(page.getByRole('button', { name: 'Show interest' })).toBeVisible();
  expect(interestSignals).toBe(1);
});

test('rejects non-image uploads before generation', async ({ page }, testInfo) => {
  await page.goto('/');
  const invalidFile = await writeTextFixture(testInfo, 'not-image.txt', 'not an image');

  await page.setInputFiles('#person-upload', invalidFile);
  await expect(page.getByText('Choose an image file')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Generate try-on' })).toBeDisabled();
});

for (const width of [320, 375, 414, 768]) {
  test(`has no horizontal overflow at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    await page.goto('/');

    const hasOverflow = await page.evaluate(() => (
      document.documentElement.scrollWidth > document.documentElement.clientWidth
    ));

    expect(hasOverflow).toBe(false);
  });
}

async function mockTryOn(page: Page, status: number, body: Record<string, unknown>, delayMs = 0) {
  await page.route('**/api/try-on', async (route) => {
    const requestBody = route.request().postDataJSON() as Record<string, unknown>;
    expect(requestBody.userImage).toEqual(expect.stringContaining('data:image/jpeg;base64,'));
    expect(requestBody.clothingImage).toEqual(expect.stringContaining('data:image/jpeg;base64,'));
    expect(requestBody.modelId).toBeTruthy();

    if (delayMs > 0) {
      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }

    await route.fulfill({
      status,
      contentType: 'application/json',
      body: JSON.stringify(body),
    });
  });
}

async function uploadFixturePair(page: Page, testInfo: TestInfo) {
  const person = await writePngFixture(testInfo, 'person.png');
  const garment = await writePngFixture(testInfo, 'garment.png');

  await page.setInputFiles('#person-upload', person);
  await page.setInputFiles('#garment-upload', garment);
}

async function writePngFixture(testInfo: TestInfo, name: string) {
  const filePath = testInfo.outputPath(name);
  await mkdir(path.dirname(filePath), { recursive: true });
  await writeFile(filePath, Buffer.from(fixtureBase64, 'base64'));
  return filePath;
}

async function writeTextFixture(testInfo: TestInfo, name: string, content: string) {
  const filePath = testInfo.outputPath(name);
  await mkdir(path.dirname(filePath), { recursive: true });
  await writeFile(filePath, content);
  return filePath;
}
