const { test, expect } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

const datasetPath = path.resolve(__dirname, '..', 'examples', 'data', 'predictive_maintenance.csv');

test.setTimeout(240000);

test('dl monitor appends log lines while training is still running', async ({ page, request }) => {
  const uploadResponse = await request.post('http://127.0.0.1:8000/api/data/upload', {
    multipart: {
      file: fs.createReadStream(datasetPath),
    },
  });
  expect(uploadResponse.ok()).toBeTruthy();
  const dataset = await uploadResponse.json();

  const startResponse = await request.post('http://127.0.0.1:8000/api/dl/train', {
    data: {
      dataset_id: dataset.id,
      target_column: 'Target',
      model_type: 'mlp_dl',
      task_type: 'classification',
      arch_config: {
        hidden_layers: [64, 32],
        dropout: 0.1,
        batch_norm: false,
      },
      opt_config: {
        optimizer: 'adam',
        learning_rate: 0.001,
      },
      train_config: {
        epochs: 100,
        batch_size: 32,
        early_stopping_patience: 100,
      },
    },
  });
  expect(startResponse.status()).toBe(201);
  const task = await startResponse.json();

  await page.goto(`http://127.0.0.1:3000/dl/monitor?taskId=${task.id}`);
  await expect(page.getByTestId('dl-log-panel')).toBeVisible({ timeout: 30000 });

  await page.waitForTimeout(1000);
  const initialCount = await page.getByTestId('dl-log-entry').count();

  await expect.poll(
    async () => page.getByTestId('dl-log-entry').count(),
    {
      timeout: 30000,
      intervals: [1000, 2000, 5000],
    }
  ).toBeGreaterThan(initialCount);

  await expect.poll(
    async () => page.getByTestId('dl-log-entry').allInnerTexts(),
    {
      timeout: 30000,
      intervals: [1000, 2000, 5000],
    }
  ).toEqual(expect.arrayContaining([
    expect.stringMatching(/val_loss=/),
  ]));
});
