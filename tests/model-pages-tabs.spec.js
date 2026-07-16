const { test, expect } = require('@playwright/test');

test('model management keeps ml dl universal tabs', async ({ page }) => {
  await page.goto('http://127.0.0.1:3000/models');

  await expect(page.getByRole('tab', { name: '机器学习模型' })).toBeVisible();
  await expect(page.getByRole('tab', { name: '深度学习模型' })).toBeVisible();
  await expect(page.getByRole('tab', { name: '通用模型' })).toBeVisible();
});

test('model deploy keeps ml dl universal tabs', async ({ page }) => {
  await page.goto('http://127.0.0.1:3000/deploy');

  await expect(page.getByRole('tab', { name: '机器学习' })).toBeVisible();
  await expect(page.getByRole('tab', { name: '深度学习' })).toBeVisible();
  await expect(page.getByRole('tab', { name: 'TimesFM / Chronos' })).toBeVisible();
});
