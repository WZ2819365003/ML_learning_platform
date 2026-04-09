const { test, expect } = require('@playwright/test');
const path = require('path');

const datasetPath = path.resolve(__dirname, '..', 'examples', 'data', 'predictive_maintenance.csv');
const samplePayload = JSON.stringify(
  [
    {
      UDI: 1,
      'Product ID': 'M14860',
      Type: 'M',
      'Air temperature [K]': 298.1,
      'Process temperature [K]': 308.6,
      'Rotational speed [rpm]': 1551,
      'Torque [Nm]': 42.8,
      'Tool wear [min]': 0,
    },
  ],
  null,
  2,
);

test.setTimeout(180000);

test('dashboard, visualization, and model management use real training data', async ({ page, request }) => {
  await page.goto('http://127.0.0.1:3000/data');

  await page.locator('input[type="file"]').setInputFiles(datasetPath);
  await expect(page.getByText('predictive_maintenance.csv').first()).toBeVisible({ timeout: 30000 });

  await page.getByTestId('configure-training-button').click();
  await expect(page).toHaveURL(/\/training\/config/);

  await page.getByTestId('start-training-button').click();
  await expect(page).toHaveURL(/\/training\/monitor/);
  const taskId = new URL(page.url()).searchParams.get('taskId');
  await expect.poll(
    async () => {
      const response = await request.get(`http://127.0.0.1:8000/api/training/${taskId}/status`);
      const payload = await response.json();
      return payload.status;
    },
    {
      timeout: 120000,
      intervals: [1000, 2000, 5000],
    }
  ).toBe('SUCCESS');

  await page.goto('http://127.0.0.1:3000/dashboard');
  await expect(page.getByTestId('dashboard-dataset-name').first()).toContainText('predictive_maintenance.csv', { timeout: 30000 });
  await expect(page.getByTestId('dashboard-model-name').first()).toContainText('random_forest', { timeout: 30000 });

  await page.goto('http://127.0.0.1:3000/results');
  await expect(page.getByTestId('results-task-select')).toContainText('predictive_maintenance.csv', { timeout: 30000 });
  await expect(page.getByTestId('confusion-matrix-chart')).toBeVisible({ timeout: 30000 });
  await expect(page.getByTestId('roc-curve-chart')).toBeVisible({ timeout: 30000 });
  await expect(page.getByTestId('feature-importance-chart')).toBeVisible({ timeout: 30000 });
  await expect(page.getByTestId('learning-curve-chart')).toBeVisible({ timeout: 30000 });

  await page.goto('http://127.0.0.1:3000/models');
  await expect(page.getByText('predictive_maintenance.csv').first()).toBeVisible({ timeout: 30000 });
  await page.getByRole('button', { name: '详情' }).first().click();
  await expect(page.getByTestId('prediction-url')).toContainText('/api/models/', { timeout: 30000 });
  await page.getByTestId('prediction-payload').fill(samplePayload);
  await page.getByTestId('run-prediction-button').click();
  await expect(page.getByTestId('prediction-result')).toContainText('prediction', { timeout: 30000 });
});
