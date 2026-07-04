import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

test('Testing poison ratio slider updates projected accuracy and verify payload', async ({ page }) => {
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = path.dirname(__filename);
  const fixturesDir = path.resolve(__dirname, 'fixtures');

  const scan = JSON.parse(fs.readFileSync(path.join(fixturesDir, 'scan.json'), 'utf-8'));
  const health = JSON.parse(fs.readFileSync(path.join(fixturesDir, 'health.json'), 'utf-8'));

  await page.route('**/api/health', async (route) => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(health) });
  });

  await page.route('**/api/scans/style-lock-scan', async (route) => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(scan) });
  });

  let lastVerifyBody: any = null;
  await page.route('**/api/scans/style-lock-scan/verify', async (route) => {
    const req = route.request();
    try {
      lastVerifyBody = req.postDataJSON();
    } catch {
      lastVerifyBody = req.postData();
    }
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        record: scan.record,
        verification: { verdict: 'OK', confidence: 'High', poisoned: false },
      }),
    });
  });

  await page.goto('/testing?scan_id=style-lock-scan');

  const impactPanel = page.getByText('Projected Accuracy').locator('..');
  const projected = impactPanel.locator('p.text-3xl');
  await expect(projected).toBeVisible();

  const slider = page.getByText('Poison Ratio', { exact: true }).locator('..').locator('..').locator('input[type="range"]');
  const displayedRatio = page.getByText('Poison Ratio', { exact: true }).locator('..').locator('span').nth(1);

  const box = await slider.boundingBox();
  expect(box).toBeTruthy();
  const y = box!.height / 2;

  const projectedBefore = await projected.textContent();

  await slider.click({ position: { x: 1, y }, force: true });
  await expect(slider).toHaveValue('1');
  await expect(displayedRatio).toHaveText(/1%/);
  const projectedAt1 = await projected.textContent();

  await slider.click({ position: { x: box!.width - 1, y }, force: true });
  await expect(slider).toHaveValue('20');
  await expect(displayedRatio).toHaveText(/20%/);
  const projectedAt20 = await projected.textContent();

  expect(projectedBefore).not.toEqual(projectedAt1);
  expect(projectedAt1).not.toEqual(projectedAt20);

  await page.getByRole('button', { name: 'Run Verification' }).click();
  await expect(page.getByText('Verdict')).toBeVisible();
  expect(lastVerifyBody).toBeTruthy();
  expect(Number(lastVerifyBody.prediction_contamination)).toBeCloseTo(0.2, 5);
});
