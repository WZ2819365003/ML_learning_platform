const { test, expect } = require('@playwright/test');
const path = require('path');

const datasetPath = path.resolve(__dirname, '..', 'examples', 'data', 'predictive_maintenance.csv');

test('upload dataset and start a training task from the frontend', async ({ page }) => {
  await page.goto('http://127.0.0.1:3000/data');

  await page.locator('input[type="file"]').setInputFiles(datasetPath);
  await expect(page.getByText('predictive_maintenance.csv').first()).toBeVisible({ timeout: 30000 });

  await page.getByTestId('configure-training-button').click();
  await expect(page).toHaveURL(/\/training\/config/);

  await expect(page.getByTestId('dataset-select')).toContainText('predictive_maintenance.csv');
  await expect(page.getByTestId('target-select')).toContainText('Target');
  await expect(page.getByTestId('model-select')).toContainText('random_forest');

  await page.getByTestId('start-training-button').click();

  await expect(page).toHaveURL(/\/training\/monitor/);
  await expect(page.getByText('predictive_maintenance.csv').first()).toBeVisible({ timeout: 10000 });
  await expect(page.getByText('random_forest').first()).toBeVisible({ timeout: 10000 });
  await expect(page.getByText('SUCCESS').first()).toBeVisible({ timeout: 60000 });
});
