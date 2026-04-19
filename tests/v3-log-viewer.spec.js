// Capture the LogViewer — both the live-tail state and a paginated view with
// multiple entries. We dispatch a new batch mid-test so the WS stream is active.
const { test, expect } = require('@playwright/test');
const BASE = 'http://127.0.0.1:3000';
const API = 'http://127.0.0.1:8000';
const TASK_ID = '9f5ce8a1-80b1-4b9d-a1c6-fe4fce39725e';

test('run inspector — logs tab (paginated to run with logs)', async ({ page }) => {
  test.setTimeout(60_000);
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto(`${BASE}/v3/tasks/${TASK_ID}`);
  await page.getByRole('tab', { name: /Run 对比/ }).click();
  await page.waitForLoadState('networkidle');

  // Go to page 2 of the leaderboard — our previous log-test run is at rank 11
  await page.locator('.ant-pagination-item-2').click().catch(() => {});
  await page.waitForTimeout(400);

  // Click 详情 on the first row of page 2 (the log-test run)
  const firstDetailBtn = page.getByRole('button', { name: /详情/ }).first();
  await firstDetailBtn.click();
  await expect(page.getByText('Run 诊断')).toBeVisible({ timeout: 5000 });
  await page.getByRole('tab', { name: /日志/ }).click();
  await page.waitForTimeout(1500);
  await page.screenshot({ path: 'screenshots/v3-audit/09-log-viewer-historical.png', fullPage: true });
});
