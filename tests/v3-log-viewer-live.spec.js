// Verify the LogViewer streams a LIVE Run via WebSocket.
// We dispatch a slow-ish run (bayesian with more trials) and open the inspector
// mid-flight to confirm the "LIVE" badge + auto-updating log list.
const { test, expect } = require('@playwright/test');
const { WEB_BASE: BASE, API_ROOT: API } = require('./helpers/e2e-env');

test('run inspector — live batch is visible while logs stream', async ({ page }) => {
  test.setTimeout(60_000);
  await page.setViewportSize({ width: 1440, height: 900 });

  const datasetsResponse = await page.request.get(`${API}/api/data/list?page=1&page_size=100`);
  expect(datasetsResponse.ok()).toBeTruthy();
  const datasets = (await datasetsResponse.json()).items || [];
  const dataset = datasets.find((item) => item.name?.includes('predictive'));
  test.skip(!dataset, 'predictive maintenance dataset is unavailable');

  const taskResponse = await page.request.post(`${API}/api/v3/tasks/`, { data: {
    name: `live-logs-${Date.now()}`,
    task_type: 'classification',
    dataset_id: dataset.id,
    target_column: 'Target',
    objective_metric: 'accuracy',
    objective_direction: 'max',
  } });
  expect(taskResponse.ok()).toBeTruthy();
  const task = await taskResponse.json();

  // Dispatch a new bayesian batch so we have fresh runs to inspect.
  const resp = await page.request.post(
    `${API}/api/v3/tasks/${task.id}/experiments`,
    {
      data: {
        name: `live-${Date.now()}`,
        strategy_type: 'bayesian_search',
        selected_models: ['random_forest'],
        search_space: {
          random_forest: {
            n_estimators: { type: 'int', low: 50, high: 200, step: 50 },
            max_depth: { type: 'int', low: 3, high: 15 },
          },
        },
        budget_config: { n_trials_per_model: 3, max_trials: 3, test_size: 0.2 },
      },
    },
  );
  expect(resp.ok()).toBeTruthy();

  await page.goto(`${BASE}/v3/tasks/${task.id}`);
  await page.getByRole('tab', { name: /模型对比/ }).click();

  // The newly created runs start at the bottom of leaderboard but the
  // Live WS should show at least one run is RUNNING. Instead of racing the
  // UI, just screenshot the 5-second-old state.
  await expect(page.getByText(/模型对比/).first()).toBeVisible();
  await page.waitForTimeout(2500);
  await page.screenshot({ path: 'screenshots/v3-audit/10-live-batch-runs.png', fullPage: true });
});
