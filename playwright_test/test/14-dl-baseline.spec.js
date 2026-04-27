// 14 — DL release gate. After v3.2.6 added torch==2.6.0+cpu to the backend
// image, every plain `import torch` in dl_models/* now works. This spec is
// the regression guard so a future requirements.txt edit that drops torch
// can never silently break /dl/train and the V3 mixed/DL flows again.
//
// We hit BOTH paths the user touches:
//   (a) standalone /api/dl/train  → poll status until SUCCESS, assert metrics
//   (b) V3 plan→task→experiments with a DL token → assert run SUCCESS
//
// Failures here = DL really broken. Don't relax assertions.
const { test, expect } = require('@playwright/test');
const { getJson, postJson } = require('../helpers/api');
const { runBaselineFlow } = require('../helpers/v3-flow');

test.describe.configure({ mode: 'serial' });

const POLL_INTERVAL = 3000;
const MAX_WAIT_MS = 90_000;
const MODEL_TOKEN = 'mlp_dl';
const TARGET_COLUMN = 'Target';

async function pickPredictiveDataset(request) {
  const r = await getJson(request, '/data/list?page=1&page_size=50');
  const ds = (r.body?.items || []).find(d => d.name === 'predictive_maintenance.csv');
  expect(ds, 'predictive_maintenance.csv must be seeded').toBeTruthy();
  return ds;
}

async function pollDlStatus(request, taskId) {
  const t0 = Date.now();
  while (Date.now() - t0 < MAX_WAIT_MS) {
    const s = await getJson(request, `/dl/${taskId}/status`);
    expect(s.ok, `status fetch failed: ${s.status} ${s.raw?.slice(0, 300)}`).toBeTruthy();
    const status = String(s.body?.status || '').toUpperCase();
    if (['SUCCESS', 'FAILED', 'CANCELED'].includes(status)) return s.body;
    await new Promise(res => setTimeout(res, POLL_INTERVAL));
  }
  throw new Error(`DL task ${taskId} did not finish within ${MAX_WAIT_MS}ms`);
}

test.describe('14 DL release gate', () => {
  test('14.1 standalone /api/dl/train mlp_dl reaches SUCCESS with metrics', async ({ request }) => {
    test.setTimeout(150_000);

    const ds = await pickPredictiveDataset(request);
    const create = await postJson(request, '/dl/train', {
      dataset_id: ds.id,
      target_column: TARGET_COLUMN,
      model_type: MODEL_TOKEN,
      task_type: 'classification',
      name: `_gate_dl_${Date.now()}`,
      train_config: { epochs: 3, batch_size: 64 },
    });
    expect(create.ok, `dl/train create: ${create.status} ${create.raw?.slice(0, 400)}`).toBeTruthy();
    expect(create.body?.id).toBeTruthy();

    const final = await pollDlStatus(request, create.body.id);
    expect(String(final.status).toUpperCase(), `final status: ${JSON.stringify(final).slice(0, 800)}`).toBe('SUCCESS');

    // Real metrics must be present — the v3.2.5 image (no torch) failed at
    // import time before any metrics could land. Guard against silent regress.
    const metrics = final.result_metrics || final.metrics || {};
    expect(metrics, 'no result_metrics returned').toBeTruthy();
    expect(metrics.history, 'history array missing — trainer did not run').toBeTruthy();
    expect(Array.isArray(metrics.history) ? metrics.history.length : 0).toBeGreaterThan(0);
    expect(typeof metrics.val_acc, 'val_acc missing from metrics').toBe('number');
  });

  test('14.2 V3 baseline with mlp_dl token completes one run end-to-end', async ({ request }) => {
    test.setTimeout(180_000);

    // runBaselineFlow normally uses logistic_regression — override to mlp_dl
    // and tell it model_family='dl' so tuning_service routes to dl_service.
    const { run } = await runBaselineFlow(request, {
      modelToken: MODEL_TOKEN,
      namePrefix: 'gate-dl',
      // bayesian/grid both 422 with DL tokens (Phase 1); baseline only.
      // runBaselineFlow already forces strategy_type='baseline'.
    });

    expect(run.model_type || run.params?.model_type, 'run.model_type mismatch').toBe(MODEL_TOKEN);
    // Some DL backends emit accuracy under different keys; just assert
    // *something* numeric so we know the trainer reached the metric stage.
    const obj = run.objective_value;
    expect(typeof obj === 'number' && Number.isFinite(obj),
      `objective_value missing: ${JSON.stringify(run).slice(0, 500)}`).toBeTruthy();
  });
});
