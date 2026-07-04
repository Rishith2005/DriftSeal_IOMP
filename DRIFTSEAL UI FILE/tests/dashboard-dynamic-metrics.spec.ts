import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

test('Dashboard metrics update when a different scan is selected', async ({ page }) => {
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = path.dirname(__filename);
  const fixturesDir = path.resolve(__dirname, 'fixtures');

  const scanA = JSON.parse(fs.readFileSync(path.join(fixturesDir, 'scan.json'), 'utf-8'));
  const scanB = JSON.parse(fs.readFileSync(path.join(fixturesDir, 'scan_b.json'), 'utf-8'));
  const health = JSON.parse(fs.readFileSync(path.join(fixturesDir, 'health.json'), 'utf-8'));

  await page.route('**/api/health', async (route) => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(health) });
  });

  await page.route('**/api/scans/**', async (route) => {
    const url = route.request().url();
    const isB = url.includes('style-lock-scan-b');
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(isB ? scanB : scanA) });
  });

  const precisionCell = () => page.locator('table tbody tr', { hasText: 'Precision' }).locator('td').nth(1);

  await page.goto('/dashboard?scan_id=style-lock-scan');
  await expect(precisionCell()).toContainText('0.87');

  await page.goto('/dashboard?scan_id=style-lock-scan-b');
  await expect(precisionCell()).toContainText('0.93');
});

