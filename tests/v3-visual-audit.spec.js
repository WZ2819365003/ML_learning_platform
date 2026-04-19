// V3 Visual Audit — capture screenshots of every surface so we can eyeball
// layout regressions. Runs against the existing E2E task with 10 completed runs.
// Skip-safe: if the task doesn't exist yet the test just skips.

const { test, expect } = require('@playwright/test');

const BASE = 'http://127.0.0.1:3000';
const TASK_ID = '9f5ce8a1-80b1-4b9d-a1c6-fe4fce39725e';
const OUT = 'screenshots/v3-audit';

test.describe('V3 Visual Audit', () => {
  test.use({ viewport: { width: 1440, height: 900 } });

  test('workbench list', async ({ page }) => {
    await page.goto(`${BASE}/v3/tasks`);
    await expect(page.getByText('建模任务工作台')).toBeVisible();
    await page.waitForLoadState('networkidle');
    await page.screenshot({ path: `${OUT}/01-list.png`, fullPage: true });
  });

  test('create task modal', async ({ page }) => {
    await page.goto(`${BASE}/v3/tasks`);
    await page.getByRole('button', { name: /新建建模任务/ }).click();
    await expect(page.getByRole('dialog')).toBeVisible();
    await page.waitForTimeout(400);
    await page.screenshot({ path: `${OUT}/02-create-modal.png`, fullPage: true });
  });

  test('task detail — overview', async ({ page }) => {
    const resp = await page.request.get(`http://127.0.0.1:8000/api/v3/tasks/${TASK_ID}`);
    if (!resp.ok()) test.skip(true, 'E2E task not present');
    await page.goto(`${BASE}/v3/tasks/${TASK_ID}`);
    await expect(page.getByText('任务信息')).toBeVisible({ timeout: 10000 });
    await page.waitForLoadState('networkidle');
    await page.screenshot({ path: `${OUT}/03-detail-overview.png`, fullPage: true });
  });

  test('task detail — experiments', async ({ page }) => {
    const resp = await page.request.get(`http://127.0.0.1:8000/api/v3/tasks/${TASK_ID}`);
    if (!resp.ok()) test.skip(true, 'E2E task not present');
    await page.goto(`${BASE}/v3/tasks/${TASK_ID}`);
    await page.getByRole('tab', { name: /实验编排/ }).click();
    await page.waitForTimeout(800);
    await page.screenshot({ path: `${OUT}/04-detail-experiments.png`, fullPage: true });
  });

  test('task detail — runs leaderboard', async ({ page }) => {
    const resp = await page.request.get(`http://127.0.0.1:8000/api/v3/tasks/${TASK_ID}`);
    if (!resp.ok()) test.skip(true, 'E2E task not present');
    await page.goto(`${BASE}/v3/tasks/${TASK_ID}`);
    await page.getByRole('tab', { name: /Run 对比/ }).click();
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(500);
    await page.screenshot({ path: `${OUT}/05-detail-runs.png`, fullPage: true });
  });

  test('task detail — explain', async ({ page }) => {
    const resp = await page.request.get(`http://127.0.0.1:8000/api/v3/tasks/${TASK_ID}`);
    if (!resp.ok()) test.skip(true, 'E2E task not present');
    await page.goto(`${BASE}/v3/tasks/${TASK_ID}`);
    await page.getByRole('tab', { name: /模型解释/ }).click();
    await page.waitForTimeout(500);
    await page.screenshot({ path: `${OUT}/06-detail-explain.png`, fullPage: true });
  });

  test('batch modal — bayesian', async ({ page }) => {
    const resp = await page.request.get(`http://127.0.0.1:8000/api/v3/tasks/${TASK_ID}`);
    if (!resp.ok()) test.skip(true, 'E2E task not present');
    await page.goto(`${BASE}/v3/tasks/${TASK_ID}`);
    await page.getByRole('tab', { name: /实验编排/ }).click();
    await page.getByRole('button', { name: /启动新批次/ }).first().click();
    const dialog = page.getByRole('dialog').filter({ hasText: '启动新的实验批次' });
    await expect(dialog).toBeVisible();
    await dialog.locator('label.ant-radio-button-wrapper').filter({ hasText: '贝叶斯' }).click();
    await page.waitForTimeout(600);
    await page.screenshot({ path: `${OUT}/07-batch-bayesian.png`, fullPage: true });
  });

  test('run inspector drawer', async ({ page }) => {
    const resp = await page.request.get(`http://127.0.0.1:8000/api/v3/tasks/${TASK_ID}`);
    if (!resp.ok()) test.skip(true, 'E2E task not present');
    await page.goto(`${BASE}/v3/tasks/${TASK_ID}`);
    await page.getByRole('tab', { name: /Run 对比/ }).click();
    await page.waitForLoadState('networkidle');
    const detailBtn = page.getByRole('button', { name: /详情/ }).first();
    if (await detailBtn.isVisible().catch(() => false)) {
      await detailBtn.click();
      await expect(page.getByText('Run 诊断')).toBeVisible({ timeout: 5000 });
      await page.waitForTimeout(500);
      await page.screenshot({ path: `${OUT}/08-run-inspector.png`, fullPage: true });
    } else {
      test.skip(true, 'no runs available');
    }
  });
});
