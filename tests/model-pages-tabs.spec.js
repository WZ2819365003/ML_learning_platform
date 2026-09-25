const { test, expect } = require('@playwright/test');

test('model management keeps ml dl and time-series tabs', async ({ page }) => {
  await page.goto('/models');

  await expect(page.getByRole('tab', { name: '机器学习模型' })).toBeVisible();
  await expect(page.getByRole('tab', { name: '深度学习模型' })).toBeVisible();
  await expect(page.getByRole('tab', { name: '时序预测' })).toBeVisible();
});

test('model deploy keeps ml dl universal tabs', async ({ page }) => {
  await page.goto('/deploy');

  await expect(page.getByRole('tab', { name: '机器学习' })).toBeVisible();
  await expect(page.getByRole('tab', { name: '深度学习' })).toBeVisible();
  await expect(page.getByRole('tab', { name: 'TimesFM / Chronos' })).toBeVisible();
});
