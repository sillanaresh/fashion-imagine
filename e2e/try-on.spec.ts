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

async function mockTryOn(page: Page, status: number, body: Record<string, unknown>) {
  await page.route('**/api/try-on', async (route) => {
    const requestBody = route.request().postDataJSON() as Record<string, unknown>;
    expect(requestBody.userImage).toEqual(expect.stringContaining('data:image/jpeg;base64,'));
    expect(requestBody.clothingImage).toEqual(expect.stringContaining('data:image/jpeg;base64,'));
    expect(requestBody.modelId).toBeTruthy();

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
