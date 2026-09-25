const { test, expect } = require('@playwright/test');

test('time-series create page renders core controls', async ({ page }) => {
  await page.goto('/ts/tasks/new');

  const dialog = page.getByRole('dialog', { name: /新建时序预测任务/ });
  await expect(dialog).toBeVisible();
  await expect(dialog.getByText('选择数据')).toBeVisible();
  await expect(dialog.getByRole('button', { name: '下一步' })).toBeVisible();
});

test('deploy page exposes TimesFM deployment tab', async ({ page }) => {
  await page.goto('/deploy');

  await page.getByRole('tab', { name: 'TimesFM / Chronos' }).click();
  await expect(page.getByText(/TimesFM.*部署|Chronos.*部署/).first()).toBeVisible();
  await expect(page.getByRole('button', { name: /新增部署|新建部署/ })).toBeVisible();
});
