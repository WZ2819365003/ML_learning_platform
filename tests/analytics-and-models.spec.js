const { test, expect } = require('@playwright/test');
const path = require('path');
const { WEB_BASE: BASE, API_ROOT: API } = require('./helpers/e2e-env');

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
  await page.goto(`${BASE}/data`);

  await page.locator('input[type="file"]').setInputFiles(datasetPath);
  await expect(page.getByText('predictive_maintenance.csv').first()).toBeVisible({ timeout: 30000 });

  const datasetsResponse = await request.get(`${API}/api/data/list?page=1&page_size=100`);
  expect(datasetsResponse.ok()).toBeTruthy();
  const dataset = ((await datasetsResponse.json()).items || [])
    .find((item) => item.name?.includes('predictive_maintenance'));
  expect(dataset).toBeTruthy();

  const startResponse = await request.post(`${API}/api/training/start`, { data: {
    dataset_id: dataset.id,
    target_column: 'Target',
    model_type: 'random_forest',
    hyperparameters: { n_estimators: 80, random_state: 42 },
    test_size: 0.2,
    eval_metrics: ['accuracy', 'f1', 'roc_auc'],
    cross_validation: { enabled: true, folds: 3 },
  } });
  expect(startResponse.status()).toBe(201);
  const taskId = (await startResponse.json()).id;
  await expect.poll(
    async () => {
      try {
        const response = await request.get(`${API}/api/training/${taskId}/status`);
        if (!response.ok()) return `HTTP_${response.status()}`;
        const payload = await response.json();
        return payload.status;
      } catch {
        return 'REQUEST_FAILED';
      }
    },
    {
      timeout: 120000,
      intervals: [1000, 2000, 5000],
    }
  ).toBe('SUCCESS');

  await page.goto(`${BASE}/dashboard`);
  await expect(page.getByText('最新数据集')).toBeVisible();
  await expect(page.getByText('predictive_maintenance.csv', { exact: false }).first()).toBeVisible({ timeout: 30000 });
  await expect(page.getByText('最近训练任务')).toBeVisible();
  await expect(page.getByText('random_forest', { exact: true }).first()).toBeVisible({ timeout: 30000 });

  await page.goto(`${BASE}/training/results?taskId=${taskId}`);
  await expect(page.getByText('结果可视化详情')).toBeVisible({ timeout: 30000 });
  await expect(page.getByText('predictive_maintenance.csv', { exact: false }).first()).toBeVisible();

  const confusionCard = page.getByText('混淆矩阵', { exact: true })
    .locator('xpath=ancestor::div[contains(concat(" ", normalize-space(@class), " "), " ant-card ")][1]');
  const rocCard = page.getByText('ROC 曲线', { exact: true })
    .locator('xpath=ancestor::div[contains(concat(" ", normalize-space(@class), " "), " ant-card ")][1]');
  await expect(confusionCard.locator('canvas').first()).toBeVisible({ timeout: 30000 });
  await expect(rocCard.locator('canvas').first()).toBeVisible({ timeout: 30000 });

  await page.getByRole('tab', { name: '训练过程' }).click();
  const cvCard = page.getByText(/分类 K-Fold 交叉验证/)
    .locator('xpath=ancestor::div[contains(concat(" ", normalize-space(@class), " "), " ant-card ")][1]');
  await expect(cvCard.locator('canvas').first()).toBeVisible({ timeout: 30000 });

  await page.getByRole('tab', { name: '解释性' }).click();
  const importanceCard = page.getByText(/模型原生特征重要性/)
    .locator('xpath=ancestor::div[contains(concat(" ", normalize-space(@class), " "), " ant-card ")][1]');
  await expect(importanceCard.locator('canvas').first()).toBeVisible({ timeout: 30000 });

  await page.goto(`${BASE}/models`);
  await expect(page.getByText('predictive_maintenance.csv').first()).toBeVisible({ timeout: 30000 });
  await page.getByRole('button', { name: '详情' }).first().click();
  const detailDialog = page.getByRole('dialog', { name: /模型详情/ });
  await expect(detailDialog.locator('code').filter({ hasText: '/api/models/' }).first()).toBeVisible({ timeout: 30000 });
  await detailDialog.locator('textarea').fill(samplePayload);
  await detailDialog.getByRole('button', { name: '运行预测' }).click();
  await expect(detailDialog.locator('pre')).toContainText('prediction', { timeout: 30000 });
});
