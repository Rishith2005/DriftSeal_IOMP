import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

test('Dashboard results UI style stays stable', async ({ page }) => {
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = path.dirname(__filename);
  const fixturesDir = path.resolve(__dirname, 'fixtures');
  const scan = JSON.parse(fs.readFileSync(path.join(fixturesDir, 'scan.json'), 'utf-8'));
  const health = JSON.parse(fs.readFileSync(path.join(fixturesDir, 'health.json'), 'utf-8'));

  await page.route('**/api/health', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(health),
    });
  });

  await page.route('**/api/scans/**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(scan),
    });
  });

  await page.goto('/dashboard?scan_id=style-lock-scan');

  await expect(page.getByText('Poison Detection')).toBeVisible();
  await expect(page.getByText('Behavioral Drift Matrix')).toBeVisible();
  await expect(page.getByText('Autoencoder')).toBeVisible();
  await expect(page.getByText('Isolation Forest')).toBeVisible();
  await expect(page.getByText('One-Class SVM')).toBeVisible();
  await expect(page.getByText('AUC-ROC')).toBeVisible();
  await expect(page.getByText('Not Eligible').first()).toBeVisible();
  await expect(page.getByText('Drift Score:', { exact: false })).toBeVisible();
  await expect(page.getByText('ML Bill of Materials & Dataset Summary', { exact: false })).toHaveCount(1);
  await expect(page.getByText('Low Risk', { exact: false })).toHaveCount(0);

  await expect(page.locator('main')).toHaveScreenshot('dashboard-results.png', {
    fullPage: false,
  });
});
