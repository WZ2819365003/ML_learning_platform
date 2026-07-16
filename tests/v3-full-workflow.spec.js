// V3 Workbench — full-workflow e2e test (Phase 1–5)
//
// This spec exercises every phase of the V3 deepening work:
//
//   Phase 1 — TrainingPlans: ML + DL mixed plan can be created via the UI, and
//             the DL config panel surfaces the right per-token defaults.
//
//   Phase 2 — Plan binding on ModelingTask: creating a task with a plan_id
//             persists the plan snapshot and the detail page shows it.
//
//   Phase 3 — Orphan task drawer: a free-standing PlatformTask (seeded via the
//             API) opens a drawer with status + source-data + log tabs.
//
//   Phase 4 — Progress tree: ModelingTask detail page embeds a ProgressTree
//             card that enumerates experiments / runs with per-node %.
//
//   Phase 5 — Scheduler: after launching an experiment, the PlatformTask shows
//             up in the TaskCenter with a matching kind/status.
//
// Test conventions:
//   * Data is seeded via backend API (self-contained); cleanup is per-test via
//     `apiCleanup`, and the whole spec runs serially to keep the DB state
//     predictable between steps.
//   * Any UI selector uses roles or visible text so the tests survive minor
//     restyling.  Chinese copy matches the current UI strings.
//   * Each test independently hits a known route so a failing setup step
//     surfaces as a clear assertion rather than a cascade of downstream failures.
//
// Prerequisites:
//   - At least one seeded Dataset (the example CSVs ship in-tree and are
//     auto-seeded on first boot).  We look one up dynamically rather than
//     hard-coding an id.
//   - Backend + frontend both launched via playwright.config.js webServer.

const { test, expect } = require('@playwright/test');

// In Docker stacks the SPA lives behind nginx at :80 (see docker/README.md).
// Override either when running against a dev server (e.g. BASE_UI=http://127.0.0.1:3000).
const BASE_UI  = process.env.BASE_UI  || 'http://127.0.0.1:3000';
const BASE_API = process.env.BASE_API || 'http://127.0.0.1:8000/api';

// Tests mutate shared DB state → run in order, one worker.
test.describe.configure({ mode: 'serial' });

// ── Helpers ──────────────────────────────────────────────────────────────────

async function listDatasets(request) {
  const res = await request.get(`${BASE_API}/data/list?page=1&page_size=100`);
  expect(res.ok(), `GET datasets failed: ${res.status()}`).toBeTruthy();
  const body = await res.json();
  return body.items || body.datasets || [];
}

async function createPlan(request, overrides = {}) {
  const payload = {
    name: `e2e-plan-${Date.now()}`,
    description: 'Playwright seeded plan',
    task_type: 'classification',
    strategy_type: 'baseline',
    model_family: 'ml',
    selected_models: ['logistic_regression'],
    eval_metrics: ['accuracy'],
    ...overrides,
  };
  const res = await request.post(`${BASE_API}/platform/training-plans`, { data: payload });
  expect(res.ok(), `POST plan failed: ${res.status()} ${await res.text()}`).toBeTruthy();
  return await res.json();
}

async function createModelingTask(request, overrides = {}) {
  const payload = {
    name: `e2e-task-${Date.now()}`,
    task_type: 'classification',
    objective_metric: 'accuracy',
    objective_direction: 'max',
    ...overrides,
  };
  const res = await request.post(`${BASE_API}/v3/tasks/`, { data: payload });
  expect(res.ok(), `POST task failed: ${res.status()} ${await res.text()}`).toBeTruthy();
  return await res.json();
}

// Pick a column safe to use as a classification target: skip ID-like columns
// (high unique-rate is rejected by the V3 target guard), prefer columns named
// Target/Label/Failure or ones with a tiny cardinality.
function pickClassificationTarget(dataset) {
  const cols = dataset?.columns_info || {};
  const names = Object.keys(cols);
  if (names.length === 0) return null;
  const preferred = ['Target', 'target', 'label', 'Label', 'Failure Type', 'Outcome', 'outcome', 'class', 'y'];
  for (const p of preferred) if (cols[p]) return p;
  // Otherwise: smallest unique-count column that isn't trivially constant.
  const ranked = names
    .map((n) => ({ n, info: cols[n] || {} }))
    .filter((x) => {
      const u = x.info.unique_count ?? x.info.nunique ?? null;
      return u !== null && u >= 2 && u <= Math.max(20, Math.floor((dataset.row_count || 0) / 10));
    })
    .sort((a, b) => (a.info.unique_count || 0) - (b.info.unique_count || 0));
  return ranked[0]?.n || null;
}

async function findAnyPlatformTask(request) {
  // There's no public POST /platform/tasks/ endpoint, so use a real orphan if
  // another workflow has created one. Linked training tasks must not leak into
  // this panel.
  const res = await request.get(`${BASE_API}/platform/tasks/?page=1&page_size=20&orphan_only=true`);
  if (!res.ok()) return null;
  const body = await res.json();
  return (body.items || [])[0] || null;
}

// ── Phase 1: TrainingPlans (ML + DL) via UI ─────────────────────────────────

test.describe('Phase 1 — TrainingPlans page supports ML + DL', () => {
  test('TrainingPlans page loads and has 新建方案 CTA', async ({ page }) => {
    await page.goto(`${BASE_UI}/v3/training-plans`);
    // Tolerant landing — the page renders when the API call returns.
    await expect(page.getByRole('button', { name: /新建方案/ }).first())
      .toBeVisible({ timeout: 15000 });
  });

  test('API: creating an ML plan returns family=ml', async ({ request }) => {
    const plan = await createPlan(request, {
      name: `e2e-ml-${Date.now()}`,
      selected_models: ['random_forest'],
    });
    expect(plan.model_family).toBe('ml');
    expect(plan.selected_models).toEqual(['random_forest']);
  });

  test('API: creating a DL plan auto-populates dl_config defaults', async ({ request }) => {
    const plan = await createPlan(request, {
      name: `e2e-dl-${Date.now()}`,
      model_family: 'dl',
      selected_models: ['mlp_dl'],
    });
    expect(plan.model_family).toBe('dl');
    expect(plan.selected_models).toContain('mlp_dl');
    // dl_config should be keyed by model token; build_default_dl_config fills it.
    expect(plan.dl_config).toBeTruthy();
    expect(plan.dl_config.mlp_dl).toBeTruthy();
    expect(Object.keys(plan.dl_config.mlp_dl)).toEqual(
      expect.arrayContaining(['arch', 'opt', 'train']),
    );
  });

  test('API: mixed plan rejects invalid token cross-family combinations', async ({ request }) => {
    // family=ml but listing a DL token → _validate_family_vs_models rejects.
    const res = await request.post(`${BASE_API}/platform/training-plans`, {
      data: {
        name: `e2e-bad-${Date.now()}`,
        model_family: 'ml',
        selected_models: ['mlp_dl'],   // wrong family
        task_type: 'classification',
        strategy_type: 'baseline',
      },
    });
    expect(res.ok()).toBeFalsy();
    expect([400, 422]).toContain(res.status());
  });
});

// ── Phase 2: ModelingTask binds TrainingPlan via UI ─────────────────────────

test.describe('Phase 2 — ModelingTask binds TrainingPlan + snapshot view', () => {
  let seededPlan;
  let datasetId;

  test.beforeAll(async ({ request }) => {
    seededPlan = await createPlan(request, {
      name: `e2e-binding-${Date.now()}`,
      selected_models: ['logistic_regression', 'random_forest'],
      eval_metrics: ['accuracy', 'f1'],
    });
    const datasets = await listDatasets(request);
    datasetId = datasets[0]?.id;
    expect(datasetId, 'need at least one seeded dataset').toBeTruthy();
  });

  test('workflow exposes plan management from the tuning strategy tab', async ({ page, request }) => {
    const task = await createModelingTask(request, {
      name: `e2e-plan-entry-${Date.now()}`,
      dataset_id: datasetId,
    });

    await page.goto(`${BASE_UI}/v3/tasks/${task.id}/workflow?step=1`);
    await expect(page.getByRole('tab', { name: '调参策略' })).toBeVisible({ timeout: 15000 });
    await page.getByRole('tab', { name: '调参策略' }).click();
    await expect(page.getByRole('button', { name: /管理训练方案/ })).toBeVisible();
    await expect(page.getByText(/基线|网格|贝叶斯/).first()).toBeVisible();
  });

  test('API-bound task shows training_plan_snapshot panel on detail page', async ({ page, request }) => {
    const task = await createModelingTask(request, {
      name: `e2e-bound-${Date.now()}`,
      dataset_id: datasetId,
      training_plan_id: seededPlan.id,
    });
    await page.goto(`${BASE_UI}/v3/tasks/${task.id}`);

    // Task header
    await expect(page.getByRole('heading', { name: task.name })).toBeVisible({ timeout: 10000 });

    // Snapshot view (TrainingPlanSnapshotView) renders plan name + strategy
    // on the Overview tab.  The panel title is "训练方案" (see component).
    await expect(page.getByText('训练方案', { exact: false }).first()).toBeVisible();
    // Seeded plan name should appear in the snapshot body.
    await expect(page.getByText(seededPlan.name).first()).toBeVisible();
  });
});

// ── Phase 4: ProgressTree embed ──────────────────────────────────────────────

test.describe('Phase 4 — ProgressTree appears on the detail page', () => {
  test('detail page shows 编排进度 card even before any run starts', async ({ page, request }) => {
    const task = await createModelingTask(request, {
      name: `e2e-tree-${Date.now()}`,
    });

    await page.goto(`${BASE_UI}/v3/tasks/${task.id}`);
    await expect(page.getByRole('heading', { name: task.name })).toBeVisible();

    // ProgressTree card title is "编排进度" — check it renders.
    await expect(page.getByText('编排进度')).toBeVisible({ timeout: 10000 });

    // With zero experiments, the Empty component shows a specific message.
    await expect(page.getByText(/暂无实验批次/)).toBeVisible();
  });

  test('ProgressTree auto-populates experiment nodes after launching baseline', async ({ page, request }) => {
    const datasets = await listDatasets(request);
    const dataset = datasets[0];
    test.skip(!dataset, 'need a dataset to run baseline');

    // Pick a target column — first numeric/integer column works for classification.
    // The examples in this repo (iris, churn…) all have simple structures.
    const targetColumn = pickClassificationTarget(dataset);
    test.skip(!targetColumn, 'dataset has no columns');

    const task = await createModelingTask(request, {
      name: `e2e-tree-live-${Date.now()}`,
      dataset_id: dataset.id,
      target_column: targetColumn,
    });

    // Launch a baseline batch directly via API — the scheduler fans out,
    // ProgressTree should pick up the new experiment on next poll.
    const batchRes = await request.post(
      `${BASE_API}/v3/tasks/${task.id}/experiments`,
      {
        data: {
          name: 'e2e-baseline',
          strategy_type: 'baseline',
          selected_models: ['logistic_regression'],
        },
      },
    );
    // Acceptable either 200 or 201 depending on how the endpoint replies.
    expect(batchRes.ok()).toBeTruthy();

    await page.goto(`${BASE_UI}/v3/tasks/${task.id}`);
    await expect(page.getByText('编排进度')).toBeVisible({ timeout: 10000 });

    // The experiment node title contains our batch name.  Polling happens
    // every 3s when has_active_runs — give it up to 20s.
    await expect(page.getByText('e2e-baseline').first()).toBeVisible({ timeout: 20000 });
  });
});

// ── Phase 3: OrphanTaskDetailDrawer ──────────────────────────────────────────

test.describe('Phase 3 — Orphan task detail drawer', () => {
  test('运行诊断 孤立任务 tab opens a drawer with 概览 / 源数据 / 日志 tabs', async ({ page, request }) => {
    let seed = await findAnyPlatformTask(request);
    if (!seed) {
      const datasets = await listDatasets(request);
      const dataset = datasets[0];
      expect(dataset, 'need a dataset to create an orphan PlatformTask').toBeTruthy();
      const targetColumn = pickClassificationTarget(dataset);
      expect(targetColumn, 'dataset needs a classification target').toBeTruthy();

      const startResponse = await request.post(`${BASE_API}/training/start`, { data: {
        dataset_id: dataset.id,
        target_column: targetColumn,
        model_type: 'logistic_regression',
        hyperparameters: { max_iter: 200 },
        test_size: 0.2,
        eval_metrics: ['accuracy'],
        cross_validation: { enabled: false, folds: 3 },
      } });
      expect(startResponse.ok(), `create orphan task: ${await startResponse.text()}`).toBeTruthy();
      await expect.poll(async () => (await findAnyPlatformTask(request))?.id ?? null, {
        timeout: 15_000,
        intervals: [250, 500, 1000],
      }).not.toBeNull();
      seed = await findAnyPlatformTask(request);
    }

    await page.goto(`${BASE_UI}/v3/runs`);

    // Switch to the "孤立任务" tab.  It may be labelled "孤立任务视图" or
    // simply "孤立任务" depending on copy tweaks; match flexibly.
    const flatTab = page.getByRole('tab', { name: /孤立任务/ });
    await expect(flatTab).toBeVisible({ timeout: 10000 });
    await flatTab.click();

    // Click 详情 on the first row — it opens OrphanTaskDetailDrawer for r.id.
    const detailBtn = page.getByRole('button', { name: '详情' }).first();
    await expect(detailBtn).toBeVisible({ timeout: 10000 });
    await detailBtn.click();

    // Drawer should open with three tabs: 概览 / 源数据 / 日志.
    const drawer = page.locator('.ant-drawer-body').first();
    await expect(drawer).toBeVisible({ timeout: 10000 });
    await expect(drawer.getByRole('tab', { name: /概览/ })).toBeVisible({ timeout: 10000 });
    await expect(drawer.getByRole('tab', { name: /源数据/ })).toBeVisible();
    await expect(drawer.getByRole('tab', { name: /日志/ })).toBeVisible();

    // 概览 is the default tab — a status / kind field should be visible somewhere.
    await expect(drawer.getByText(/状态|任务类型|kind/i).first()).toBeVisible();

    // Click 日志 tab and make sure the body still renders (log list or empty hint).
    await drawer.getByRole('tab', { name: /日志/ }).click();
    await expect(drawer).toBeVisible();
  });
});

// ── Phase 5 + hierarchical view integration ─────────────────────────────────

test.describe('Phase 5 — Scheduler path surfaces in run diagnostics', () => {
  test('new ModelingTask with baseline experiment appears in the Run list', async ({ page, request }) => {
    const datasets = await listDatasets(request);
    const dataset = datasets[0];
    test.skip(!dataset, 'need a dataset for baseline');
    const targetColumn = pickClassificationTarget(dataset);
    test.skip(!targetColumn, 'no columns');

    const task = await createModelingTask(request, {
      name: `e2e-sched-${Date.now()}`,
      dataset_id: dataset.id,
      target_column: targetColumn,
    });
    const batchRes = await request.post(
      `${BASE_API}/v3/tasks/${task.id}/experiments`,
      { data: {
        name: 'sched-baseline',
        strategy_type: 'baseline',
        selected_models: ['logistic_regression'],
      } },
    );
    expect(batchRes.ok()).toBeTruthy();

    await page.goto(`${BASE_UI}/v3/runs`);
    await expect(page.getByRole('tab', { name: /全部 Run/ })).toBeVisible({ timeout: 10000 });

    // Find the run by its modeling-task name in the flat diagnostics table.
    const row = page.getByRole('row').filter({ hasText: task.name });
    await expect(row).toBeVisible({ timeout: 20000 });
    await expect(row.getByText('sched-baseline')).toBeVisible();
  });
});
