// Capture the LogViewer — both the live-tail state and a paginated view with
// multiple entries. We dispatch a new batch mid-test so the WS stream is active.
const { test, expect } = require('@playwright/test');
const BASE = process.env.BASE_UI || 'http://127.0.0.1:3000';

test('run inspector — logs tab (paginated to run with logs)', async ({ page }) => {
  test.setTimeout(60_000);
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto(`${BASE}/v3/runs`);
  await page.waitForLoadState('networkidle');

  const firstDiagnosticButton = page.getByRole('button', { name: '诊断' }).first();
  test.skip(!(await firstDiagnosticButton.isVisible().catch(() => false)), 'no runs available');
  await firstDiagnosticButton.click();
  await expect(page.getByText('Run 诊断')).toBeVisible({ timeout: 5000 });
  await page.getByRole('tab', { name: /日志/ }).click();
  await page.waitForTimeout(1500);
  await page.screenshot({ path: 'screenshots/v3-audit/09-log-viewer-historical.png', fullPage: true });
});
