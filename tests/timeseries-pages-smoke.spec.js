const { test, expect } = require('@playwright/test');

test('time-series create page renders core controls', async ({ page }) => {
  await page.goto('/ts/tasks/new');

  await expect(page.getByRole('heading', { name: '新建时序任务' })).toBeVisible();
  await expect(page.getByRole('button', { name: '管理 TimesFM 部署' })).toBeVisible();
  await expect(page.getByText('提交预测任务')).toBeVisible();
});

test('deploy page exposes TimesFM deployment tab', async ({ page }) => {
  await page.goto('/deploy');

  await page.getByRole('tab', { name: 'TimesFM 部署' }).click();
  await expect(page.getByRole('heading', { name: 'TimesFM 部署' })).toBeVisible();
  await expect(page.getByRole('button', { name: '新建部署' })).toBeVisible();
});
