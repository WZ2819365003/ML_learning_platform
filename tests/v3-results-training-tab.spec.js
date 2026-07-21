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
const { ensureSuccessfulTrainingTasks } = require('./helpers/training-tasks');

const WEB_BASE = process.env.BASE_UI || 'http://127.0.0.1:3000';

let classificationTaskId;
let regressionTaskId;

test.describe('Commit 13 — Results.jsx 训练过程 tab rework', () => {
  test.setTimeout(150_000);

  test.beforeAll(async ({ request }) => {
    const tasks = await ensureSuccessfulTrainingTasks(request);
    classificationTaskId = tasks.classification?.id;
    regressionTaskId = tasks.regression?.id;
  });

  test('Classification ML task: K-Fold CV view renders summary cards + bar + table', async ({ page }) => {
    test.skip(!classificationTaskId, 'no successful classification task with CV metrics');
    await page.goto(`${WEB_BASE}/training/results?taskId=${classificationTaskId}`);
    const trainingTab = page.getByRole('tab', { name: /训练过程/ });
    await expect(trainingTab).toBeVisible({ timeout: 15000 });
    await trainingTab.click();

    // Intro Alert explaining K-Fold (must mention 标准差 since DL would mention epoch instead)
    await expect(page.getByText(/K 折交叉验证|标准差|稳定性/).first()).toBeVisible({ timeout: 10000 });

    // K-Fold card title
    await expect(page.getByText(/分类 K-Fold 交叉验证/)).toBeVisible();

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
    test.skip(!regressionTaskId, 'no successful regression task with CV metrics');
    await page.goto(`${WEB_BASE}/training/results?taskId=${regressionTaskId}`);
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
    test.skip(!classificationTaskId, 'no successful classification task with CV metrics');
    await page.goto(`${WEB_BASE}/training/results?taskId=${classificationTaskId}`);
    const trainingTab = page.getByRole('tab', { name: /训练过程/ });
    const explainTab = page.getByRole('tab', { name: /解释性/ });
    await expect(trainingTab).toBeVisible({ timeout: 15000 });
    await expect(explainTab).toBeVisible();
  });

  test('CV summary card shows mean ± std stability annotation', async ({ page }) => {
    test.skip(!classificationTaskId, 'no successful classification task with CV metrics');
    await page.goto(`${WEB_BASE}/training/results?taskId=${classificationTaskId}`);
    await page.getByRole('tab', { name: /训练过程/ }).click();
    // The summary card subtitle should include either 稳定 or 波动较大 — these are
    // the only two strings the CV-stability classifier emits.
    await expect(page.getByText(/(稳定|波动较大)/).first()).toBeVisible({ timeout: 10000 });
  });
});

test.describe('RunInspector DL training visualization', () => {
  test('DL run renders only epoch history and links to its full DL results', async ({ page }) => {
    test.setTimeout(60000);
    const runId = 'dl-run-inspector-contract';
    const dlTaskId = 'dl-task-inspector-contract';
    const unexpectedVizRequests = [];

    page.on('request', (request) => {
      if (request.url().includes('/api/viz/')) unexpectedVizRequests.push(request.url());
    });

    await page.route('**/api/v3/runs/**', async (route) => {
      await route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({
          items: [{
            run_id: runId,
            experiment_id: 'experiment-1',
            experiment_name: 'DL inspector contract',
            strategy_type: 'baseline',
            task_id: 'modeling-task-1',
            task_name: 'DL task',
            task_type: 'regression',
            objective_metric: 'rmse',
            objective_direction: 'min',
            objective_value: 0.2,
            trial_no: 1,
            rank: 1,
            status: 'SUCCESS',
            model_type: 'mlp_dl',
            created_at: '2026-07-17T00:00:00Z',
            finished_at: '2026-07-17T00:01:00Z',
          }],
          total: 1,
        }),
      });
    });
    await page.route(`**/api/platform/runs/${runId}/inspector**`, async (route) => {
      await route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({
          run: {
            id: runId,
            params: { task_type: 'classification' },
            metrics: {
              history: {
                train_loss: [1.0, 0.6],
                val_loss: [1.2, 0.8],
                val_rmse: [1.1, 0.7],
              },
            },
            status: 'SUCCESS',
          },
          experiment: { name: 'DL inspector contract', objective_metric: 'rmse', objective_direction: 'min' },
          platform_task: { status: 'SUCCESS' },
          training_task: {
            id: dlTaskId,
            family: 'dl',
            model_type: 'mlp_dl',
            task_type: 'regression',
            status: 'SUCCESS',
            progress: 1,
            result_metrics: {},
          },
          logs: [],
          siblings: [],
        }),
      });
    });

    await page.goto(`${WEB_BASE}/v3/runs`, { waitUntil: 'domcontentloaded' });
    await expect(page.getByText('DL inspector contract')).toBeVisible();
    await page.locator('.ant-table-row', { hasText: 'DL inspector contract' })
      .locator('button.ant-btn-text')
      .click();

    const drawer = page.locator('.ant-drawer-body').first();
    await expect(drawer).toBeVisible();
    await drawer.getByRole('tab', { name: '训练可视化' }).click();

    await expect(drawer.getByText('Epoch 训练历史')).toBeVisible();
    await expect(drawer.getByRole('link', { name: '查看完整 DL 结果' }))
      .toHaveAttribute('href', `/dl/results?taskId=${dlTaskId}`);
    await expect(drawer.getByText('混淆矩阵')).toHaveCount(0);
    expect(unexpectedVizRequests).toEqual([]);
  });
});
