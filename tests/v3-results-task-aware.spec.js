// Commit 12 smoke — Results.jsx task-type-aware layout
//
// Asserts that the 4-tab roster differs by task type (no more shared
// «one-size-fits-all» tabs that show ROC/PR for regression or
// predicted-vs-actual for classification):
//
//   Classification: 性能 | 训练过程 | 解释性 | 阈值与分布
//   Regression    : 预测对比 | 训练过程 | 解释性 | 误差诊断
//
// Plus the regression «误差诊断» tab carries the new diagnostic charts
// (R²/RMSE/MAE/MAPE strip + residual histogram + Q-Q + residual scatter).

const { test, expect } = require('@playwright/test');

const WEB_BASE = 'http://127.0.0.1:3000';

// Fixtures: task IDs that the storage/models registry resolves to. If your
// local storage doesn't have these specific runs, either swap them out or
// retrain the v3.1.1 demo bundle. They're kept inline so a reader can wire
// the assertions back to the underlying data without chasing fixtures.
const CLASSIFICATION_TASK_ID = '12d406cb-769c-42df-b047-09ff79475f90'; // xgboost
const REGRESSION_TASK_ID     = 'e62e3196-aba7-4f39-8e6c-b49602bb1eb4'; // random_forest_regressor

test.describe('Commit 12 — Results.jsx task-type-aware tabs', () => {
  test('Classification task shows 性能 / 阈值与分布 tabs (no 预测对比 / 误差诊断)', async ({ page }) => {
    await page.goto(`${WEB_BASE}/training/results?taskId=${CLASSIFICATION_TASK_ID}`);
    // Wait for tab strip to render
    await expect(page.getByRole('tab', { name: /性能/ })).toBeVisible({ timeout: 15000 });
    await expect(page.getByRole('tab', { name: /训练过程/ })).toBeVisible();
    await expect(page.getByRole('tab', { name: /解释性/ })).toBeVisible();
    await expect(page.getByRole('tab', { name: /阈值与分布/ })).toBeVisible();
    // Regression-only tabs MUST be absent
    await expect(page.getByRole('tab', { name: /预测对比/ })).toHaveCount(0);
    await expect(page.getByRole('tab', { name: /误差诊断/ })).toHaveCount(0);
  });

  test('Regression task shows 预测对比 / 误差诊断 tabs (no 性能 / 阈值与分布)', async ({ page }) => {
    await page.goto(`${WEB_BASE}/training/results?taskId=${REGRESSION_TASK_ID}`);
    await expect(page.getByRole('tab', { name: /预测对比/ })).toBeVisible({ timeout: 15000 });
    await expect(page.getByRole('tab', { name: /训练过程/ })).toBeVisible();
    await expect(page.getByRole('tab', { name: /解释性/ })).toBeVisible();
    await expect(page.getByRole('tab', { name: /误差诊断/ })).toBeVisible();
    // Classification-only tabs MUST be absent
    await expect(page.getByRole('tab', { name: /^性能$|性能\s/ })).toHaveCount(0);
    await expect(page.getByRole('tab', { name: /阈值与分布/ })).toHaveCount(0);
  });

  test('Regression 预测对比 tab renders chart with y=x diagonal + ±1σ band legend', async ({ page }) => {
    await page.goto(`${WEB_BASE}/training/results?taskId=${REGRESSION_TASK_ID}`);
    const comparisonTab = page.getByRole('tab', { name: /预测对比/ });
    await expect(comparisonTab).toBeVisible({ timeout: 15000 });
    await comparisonTab.click();
    // Card description text should mention y = x and ±1σ
    await expect(page.getByText(/y = x.*理想线|±1σ/).first()).toBeVisible({ timeout: 10000 });
  });

  test('Regression 误差诊断 tab carries R²/RMSE/MAE/MAPE metric strip', async ({ page }) => {
    await page.goto(`${WEB_BASE}/training/results?taskId=${REGRESSION_TASK_ID}`);
    const diagTab = page.getByRole('tab', { name: /误差诊断/ });
    await expect(diagTab).toBeVisible({ timeout: 15000 });
    await diagTab.click();
    // The 4 stat cards
    for (const label of ['R²', 'RMSE', 'MAE', 'MAPE (%)']) {
      await expect(page.getByText(label, { exact: false }).first()).toBeVisible({ timeout: 10000 });
    }
    // Section titles for the three diagnostic plots
    await expect(page.getByText(/残差直方图/)).toBeVisible();
    await expect(page.getByText(/残差 vs 预测值/)).toBeVisible();
    await expect(page.getByText(/Q-Q 正态性检验/)).toBeVisible();
  });

  test('Switching task type resets active tab (no stale «performance» key on regression)', async ({ page }) => {
    // Simulate user navigating from a classification result to a regression one;
    // the second visit must not crash by trying to render the missing tab.
    await page.goto(`${WEB_BASE}/training/results?taskId=${CLASSIFICATION_TASK_ID}`);
    await expect(page.getByRole('tab', { name: /性能/ })).toBeVisible({ timeout: 15000 });
    await page.goto(`${WEB_BASE}/training/results?taskId=${REGRESSION_TASK_ID}`);
    await expect(page.getByRole('tab', { name: /预测对比/ })).toBeVisible({ timeout: 15000 });
    // The default-active tab on regression is 预测对比 (per the page's
    // defaultActiveKey + key={taskKind} re-keying).
    const activeTab = page.locator('.ant-tabs-tab-active').first();
    await expect(activeTab).toContainText(/预测对比/);
  });
});
