const { test, expect } = require('@playwright/test');
const path = require('path');

const datasetPath = path.resolve(__dirname, '..', 'examples', 'data', 'predictive_maintenance.csv');
const BASE = process.env.BASE_UI || 'http://127.0.0.1:3000';
const API = (process.env.BASE_API || 'http://127.0.0.1:8000').replace(/\/api\/?$/, '');

test('upload dataset and start a V3 training task from the frontend', async ({ page }) => {
  test.setTimeout(180_000);
  await page.goto(`${BASE}/v3/tasks/new/workflow`);

  await page.locator('input[type="file"]').setInputFiles(datasetPath);
  await expect(page.getByText('predictive_maintenance.csv').first()).toBeVisible({ timeout: 30000 });
  await page.getByLabel('任务名称').fill(`ui-upload-train-${Date.now()}`);

  const targetSelect = page.getByLabel('目标列');
  await expect(targetSelect).toBeEnabled({ timeout: 15000 });
  await targetSelect.click();
  await targetSelect.fill('Target');
  await page.locator('.ant-select-dropdown:visible .ant-select-item-option')
    .filter({ hasText: /^Target/ })
    .click();

  await page.getByRole('button', { name: /创建并继续/ }).click();
  await expect(page).toHaveURL(/\/v3\/tasks\/[\w-]+\/workflow/);

  const modelSelect = page.locator('.model-selector').getByRole('combobox');
  await expect(modelSelect).toBeVisible({ timeout: 15000 });
  await modelSelect.click();
  await page.locator('.ant-select-dropdown:visible')
    .getByText('随机森林', { exact: true })
    .click();
  await page.getByRole('button', { name: /启动机器学习训练/ }).click();

  await expect(page.getByText('训练进度')).toBeVisible({ timeout: 15000 });
  const taskId = page.url().match(/\/v3\/tasks\/([\w-]+)\/workflow/)?.[1];
  expect(taskId).toBeTruthy();
  await expect.poll(async () => {
    try {
      const response = await page.request.get(`${API}/api/v3/tasks/${taskId}/runs`);
      if (!response.ok()) return 'REQUEST_FAILED';
      const runs = (await response.json()).items || [];
      if (!runs.length) return 'PENDING';
      return runs.every((run) => ['SUCCESS', 'FAILED', 'CANCELED'].includes(run.status))
        ? runs.some((run) => run.status === 'SUCCESS') ? 'SUCCESS' : 'FAILED'
        : 'RUNNING';
    } catch {
      return 'REQUEST_FAILED';
    }
  }, { timeout: 120_000, intervals: [1000, 2000, 5000] }).toBe('SUCCESS');
});
