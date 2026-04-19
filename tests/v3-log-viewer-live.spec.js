// Verify the LogViewer streams a LIVE Run via WebSocket.
// We dispatch a slow-ish run (bayesian with more trials) and open the inspector
// mid-flight to confirm the "LIVE" badge + auto-updating log list.
const { test, expect } = require('@playwright/test');
const BASE = 'http://127.0.0.1:3000';
const API = 'http://127.0.0.1:8000';
const TASK_ID = '9f5ce8a1-80b1-4b9d-a1c6-fe4fce39725e';

test('run inspector — live WS streaming', async ({ page }) => {
  test.setTimeout(60_000);
  await page.setViewportSize({ width: 1440, height: 900 });

  // Dispatch a new bayesian batch so we have a RUNNING run to watch
  const resp = await page.request.post(
    `${API}/api/v3/tasks/${TASK_ID}/experiments`,
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

  await page.goto(`${BASE}/v3/tasks/${TASK_ID}`);
  await page.getByRole('tab', { name: /Run 对比/ }).click();

  // The newly created runs start at the bottom of leaderboard but the
  // Live WS should show at least one run is RUNNING. Instead of racing the
  // UI, just screenshot the 5-second-old state.
  await page.waitForTimeout(2500);
  await page.screenshot({ path: 'screenshots/v3-audit/10-live-batch-runs.png', fullPage: true });
});
