import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

test('Remediation workflow recognizes an already cured scan', async ({ page }) => {
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = path.dirname(__filename);
  const fixturesDir = path.resolve(__dirname, 'fixtures');

  const scanCured = JSON.parse(fs.readFileSync(path.join(fixturesDir, 'scan_cured.json'), 'utf-8'));
  const health = JSON.parse(fs.readFileSync(path.join(fixturesDir, 'health.json'), 'utf-8'));

  await page.route('**/api/health', async (route) => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(health) });
  });

  await page.route('**/api/scans/style-lock-scan', async (route) => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(scanCured) });
  });

  await page.goto('/remediation?scan_id=style-lock-scan');
  await expect(page.getByText('Cure Complete!')).toBeVisible();
  await expect(page.getByText('Signed PDF Report')).toBeVisible();
});

test('Remediation workflow applies a recommended fix and reaches download step', async ({ page }) => {
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = path.dirname(__filename);
  const fixturesDir = path.resolve(__dirname, 'fixtures');

  const scanBase = JSON.parse(fs.readFileSync(path.join(fixturesDir, 'scan.json'), 'utf-8'));
  const health = JSON.parse(fs.readFileSync(path.join(fixturesDir, 'health.json'), 'utf-8'));

  let scanState: any = scanBase;

  await page.route('**/api/health', async (route) => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(health) });
  });

  await page.route('**/api/scans/style-lock-scan', async (route) => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(scanState) });
  });

  await page.route('**/api/scans/style-lock-scan/recommendations', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        scan_id: 'style-lock-scan',
        poison_score: 0.78,
        severity_pct: 78,
        dataset_rows: 8100,
        default_fix_id: 0,
        fixes: [
          {
            id: 0,
            name: 'Outlier Filtering',
            type: 'Statistical Sanitization',
            metrics: { expected_f1_boost_pct: 8.2, gpu_minutes: 12, success_rate_pct: 94 },
            params: { dataset_contamination: 0.01, predictions_contamination: 0.05, retrain: false, fix_id: 0 },
          },
          {
            id: 1,
            name: 'Adversarial Retraining',
            type: 'Model Hardening',
            metrics: { expected_f1_boost_pct: 11.5, gpu_minutes: 45, success_rate_pct: 88 },
            params: { dataset_contamination: 0.01, predictions_contamination: 0.05, retrain: true, fix_id: 1 },
          },
          {
            id: 2,
            name: 'Clean Sample Boosting',
            type: 'Dataset Augmentation',
            metrics: { expected_f1_boost_pct: 6.8, gpu_minutes: 8, success_rate_pct: 91 },
            params: { dataset_contamination: 0.02, predictions_contamination: 0.05, retrain: false, fix_id: 2 },
          },
        ],
      }),
    });
  });

  await page.route('**/api/scans/style-lock-scan/cure', async (route) => {
    const req = route.request();
    const body = req.postDataJSON() as any;
    expect(body.fix_id).toBe(0);
    expect(body.retrain).toBe(false);

    const now = '2026-03-13T00:00:00Z';
    scanState = {
      ...scanBase,
      record: {
        ...scanBase.record,
        cured: true,
        cured_at: now,
        applied_fixes: [{ id: 0, applied_at: now, params: { dataset_contamination: 0.01, predictions_contamination: 0.05, retrain: false } }],
      },
      cure: {
        meta_path: scanBase.record.meta_path,
        updated: true,
        sanitized: { dataset: { csv: { rows_in: 8100, rows_out: 8019, dropped: 81, contamination: 0.01, output_path: 'T:/tmp/sanitized.csv' } } },
        retrain: null,
      },
      cure_artifacts: {
        report_json: 'T:/tmp/cure_report.json',
        report_pdf: 'T:/tmp/cure_report.pdf',
        bundle: 'T:/tmp/cured_bundle.zip',
        cured_metrics: null,
      },
    };

    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ record: scanState.record, cure: scanState.cure, cure_artifacts: scanState.cure_artifacts, verification: null }),
    });
  });

  await page.goto('/remediation?scan_id=style-lock-scan');
  await page.getByRole('button', { name: 'Continue to Recommendations' }).click();
  const apply = page.getByRole('button', { name: 'Apply Fix' });
  await expect(apply).toBeEnabled();
  await apply.click();

  await expect(page.getByText('Cure Complete!')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Retrain Model' })).toBeVisible();
});
