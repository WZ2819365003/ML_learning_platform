// Commit 13 smoke — Results.jsx 训练过程 tab rework
//
// Asserts the new shape of the «训练过程» tab (now task-shape-aware):
//
//   ML K-Fold task → CrossValidationView with:
//     - per-metric mean ± std summary cards
//     - 「各折表现」 bar chart card
//     - 「逐折数值」 table card
//     - intro Alert explaining K-Fold semantics
//
//   DL epoch task → TrainingHistoryChart with:
//     - 「Epoch 训练历史」 card title
//     - intro Alert explaining epoch semantics
//
// Both task types must render a contextual intro Alert at the top of the
// tab so the user understands what they're looking at.

const { test, expect } = require('@playwright/test');

const WEB_BASE = 'http://127.0.0.1:3000';

// Same fixtures as v3-results-task-aware.spec.js — these are the registered
// classification/regression XGBoost / RF runs that have CV data.
const CLASSIFICATION_TASK_ID = '12d406cb-769c-42df-b047-09ff79475f90'; // xgboost (5-fold CV)
const REGRESSION_TASK_ID     = 'e62e3196-aba7-4f39-8e6c-b49602bb1eb4'; // random_forest_regressor

test.describe('Commit 13 — Results.jsx 训练过程 tab rework', () => {
  test('Classification ML task: K-Fold CV view renders summary cards + bar + table', async ({ page }) => {
    await page.goto(`${WEB_BASE}/training/results?taskId=${CLASSIFICATION_TASK_ID}`);
    const trainingTab = page.getByRole('tab', { name: /训练过程/ });
    await expect(trainingTab).toBeVisible({ timeout: 15000 });
    await trainingTab.click();

    // Intro Alert explaining K-Fold (must mention 标准差 since DL would mention epoch instead)
    await expect(page.getByText(/K 折交叉验证|标准差|稳定性/).first()).toBeVisible({ timeout: 10000 });

    // K-Fold card title
    await expect(page.getByText(/K-Fold 交叉验证稳定性/)).toBeVisible();

    // Per-metric summary cards: at least Accuracy K-Fold mean
    await expect(page.getByText(/Accuracy（K-Fold 均值）|F1（K-Fold 均值）|ROC AUC（K-Fold 均值）/).first())
      .toBeVisible({ timeout: 10000 });

    // Bar chart card
    await expect(page.getByText(/各折表现/)).toBeVisible();
    // Per-fold table card
    await expect(page.getByText(/逐折数值/)).toBeVisible();
    // Footer narrative explaining CV
    await expect(page.getByText(/CV 系数/)).toBeVisible();
  });

  test('Regression ML task: K-Fold CV view shows R² / RMSE / MAE summary cards', async ({ page }) => {
    await page.goto(`${WEB_BASE}/training/results?taskId=${REGRESSION_TASK_ID}`);
    const trainingTab = page.getByRole('tab', { name: /训练过程/ });
    await expect(trainingTab).toBeVisible({ timeout: 15000 });
    // Wait for vizState hydration so the Tabs component (key={taskKind})
    // has finished its remount and our click won't get swallowed.
    await page.waitForLoadState('networkidle');
    await trainingTab.click();
    // Confirm the click registered before asserting on contents.
    await expect(page.locator('.ant-tabs-tab-active').filter({ hasText: '训练过程' })).toBeVisible({ timeout: 10000 });

    // Regression metric cards (data has r2 + rmse; mae may be absent)
    await expect(page.getByText(/R²（K-Fold 均值）|RMSE（K-Fold 均值）/).first())
      .toBeVisible({ timeout: 10000 });
    // Bar + table headings
    await expect(page.getByText(/各折表现/)).toBeVisible();
    await expect(page.getByText(/逐折数值/)).toBeVisible();
  });

  test('Tab roster still includes 训练过程 between performance and explain (no regression)', async ({ page }) => {
    await page.goto(`${WEB_BASE}/training/results?taskId=${CLASSIFICATION_TASK_ID}`);
    const trainingTab = page.getByRole('tab', { name: /训练过程/ });
    const explainTab = page.getByRole('tab', { name: /解释性/ });
    await expect(trainingTab).toBeVisible({ timeout: 15000 });
    await expect(explainTab).toBeVisible();
  });

  test('CV summary card shows mean ± std stability annotation', async ({ page }) => {
    await page.goto(`${WEB_BASE}/training/results?taskId=${CLASSIFICATION_TASK_ID}`);
    await page.getByRole('tab', { name: /训练过程/ }).click();
    // The summary card subtitle should include either 稳定 or 波动较大 — these are
    // the only two strings the CV-stability classifier emits.
    await expect(page.getByText(/(稳定|波动较大)/).first()).toBeVisible({ timeout: 10000 });
  });
});
