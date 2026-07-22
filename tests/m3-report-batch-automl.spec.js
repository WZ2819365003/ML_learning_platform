// @ts-check
// M3 end-to-end: AutoML sweep → final evaluation → task report → batch prediction.
//
// This walks the whole chain through the UI because the pieces only mean
// anything together: AutoML runs have to be *comparable* (selection metrics),
// which is what lets them be finalised, which is what unlocks the report.
// Testing them in isolation would miss exactly the integration that was broken.
const { test, expect } = require('@playwright/test');
const path = require('path');
const { API_ROOT: API } = require('./helpers/e2e-env');

const TRAIN_CSV = path.resolve(__dirname, '..', 'examples', 'data', 'telco_churn.csv');
const PREDICT_CSV = path.resolve(__dirname, '..', 'examples', 'data', 'telco_churn_to_predict.csv');

/** Poll an API predicate until it holds. Playwright's expect.poll on request. */
async function waitFor(request, url, predicate, { timeout = 180000, interval = 2000 } = {}) {
  const deadline = Date.now() + timeout;
  let last;
  while (Date.now() < deadline) {
    const resp = await request.get(url);
    last = resp.ok() ? await resp.json() : { _status: resp.status() };
    if (predicate(last)) return last;
    await new Promise((r) => setTimeout(r, interval));
  }
  throw new Error(`timed out waiting on ${url}; last payload: ${JSON.stringify(last).slice(0, 400)}`);
}

test.describe('M3 · AutoML → 最终评估 → 报告 → 批量预测', () => {
  test.describe.configure({ mode: 'serial', timeout: 300000 });

  let taskId;
  let deploymentId;

  test('AutoML 一键调优产出可比较的候选', async ({ page, request }) => {
    // --- upload dataset + create task through the API (the UI wizard is
    // covered by upload-and-train.spec.js; this test is about M3). ---
    const upload = await request.post(`${API}/api/data/upload`, {
      multipart: { file: { name: 'telco_churn.csv', mimeType: 'text/csv', buffer: require('fs').readFileSync(TRAIN_CSV) } },
    });
    expect(upload.ok()).toBeTruthy();
    const datasetId = (await upload.json()).id;

    const created = await request.post(`${API}/api/v3/tasks/`, {
      data: {
        name: 'E2E 流失预测',
        dataset_id: datasetId,
        target_column: 'churn',
        task_type: 'classification',
        objective_metric: 'accuracy',
      },
    });
    expect(created.ok()).toBeTruthy();
    taskId = (await created.json()).id;

    // --- launch AutoML from the UI ---
    await page.goto(`/v3/tasks/${taskId}`);
    await expect(page.getByRole('heading', { name: 'E2E 流失预测' })).toBeVisible();

    await page.getByRole('button', { name: /AutoML 一键调优/ }).click();
    await expect(page.getByText(/AutoML 已启动/)).toBeVisible({ timeout: 30000 });

    // --- every candidate must finish ---
    const runs = await waitFor(
      request,
      `${API}/api/v3/tasks/${taskId}/runs`,
      (d) => {
        const items = Array.isArray(d) ? d : (d.items || []);
        return items.length > 0 && items.every((r) => ['SUCCESS', 'FAILED'].includes(r.status));
      },
    );
    const items = Array.isArray(runs) ? runs : (runs.items || []);
    expect(items.length).toBeGreaterThan(1);
    expect(items.filter((r) => r.status === 'SUCCESS').length).toBeGreaterThan(0);

    // The property the M3-3 rewrite exists for: AutoML results carry selection
    // metrics, so they can be ranked against anything else and finalised.
    const board = await (await request.get(`${API}/api/v3/tasks/${taskId}/leaderboard`)).json();
    const rows = Array.isArray(board) ? board : (board.items || []);
    expect(rows.length).toBeGreaterThan(0);
    for (const row of rows) {
      const keys = Object.keys(row.metrics || {});
      expect(keys.some((k) => k.startsWith('selection_')), `run has no selection metric: ${keys}`).toBeTruthy();
    }

    // Duplicate model_types in the registry must survive as separate trials.
    const models = rows.map((r) => (r.params || {}).model_type);
    expect(new Set(models).size, 'candidates were collapsed by model_type').toBeLessThan(models.length);
  });

  test('最终评估后报告可生成且口径分离', async ({ page, request }) => {
    const finalize = await request.post(`${API}/api/v3/tasks/${taskId}/final-evaluation`);
    expect(finalize.ok()).toBeTruthy();
    expect((await finalize.json()).status).toBe('finalized');

    await page.goto(`/v3/tasks/${taskId}`);
    await page.getByRole('tab', { name: /任务报告/ }).click();

    await expect(page.getByRole('heading', { name: /建模任务报告/ })).toBeVisible({ timeout: 30000 });
    await expect(page.getByText(/在封存测试集上的/)).toBeVisible();

    // The load-bearing warning: without it a reader compares selection numbers
    // against the final number and concludes the model regressed.
    await expect(page.getByText(/不可与第 2 节的最终评估结果直接比较/)).toBeVisible();

    // …and it must be styled as a warning, not buried as body text.
    const warning = page.locator('.ant-alert-warning', { hasText: '不可与第 2 节' });
    await expect(warning).toBeVisible();

    await expect(page.getByRole('button', { name: /下载 Markdown/ })).toBeEnabled();
  });

  test('报告在未定稿任务上明确拒绝并给出下一步', async ({ page, request }) => {
    const upload = await request.post(`${API}/api/data/upload`, {
      multipart: { file: { name: 'telco_churn.csv', mimeType: 'text/csv', buffer: require('fs').readFileSync(TRAIN_CSV) } },
    });
    const dsId = (await upload.json()).id;
    const created = await request.post(`${API}/api/v3/tasks/`, {
      data: {
        name: 'E2E 未定稿', dataset_id: dsId, target_column: 'churn',
        task_type: 'classification', objective_metric: 'accuracy',
      },
    });
    const freshId = (await created.json()).id;

    await page.goto(`/v3/tasks/${freshId}`);
    await page.getByRole('tab', { name: /任务报告/ }).click();
    await expect(page.getByText(/尚未执行最终评估/)).toBeVisible({ timeout: 30000 });
  });

  test('批量预测：上传 CSV → 轮询 → 下载结果', async ({ page, request }) => {
    const detail = await (await request.get(`${API}/api/v3/tasks/${taskId}`)).json();
    const winner = (detail.final_evaluation || {}).winner_run_id;
    expect(winner, 'no winner run recorded').toBeTruthy();

    const deployed = await request.post(`${API}/api/v3/tasks/${taskId}/runs/${winner}/deploy`, {
      data: { name: 'E2E 流失预测服务', description: 'playwright' },
    });
    expect(deployed.ok()).toBeTruthy();
    deploymentId = (await deployed.json()).deployment_id;

    await page.goto('/deploy');
    await page.getByRole('button', { name: /详情/ }).first().click();
    await page.getByRole('tab', { name: /批量预测/ }).click();
    await expect(page.getByText(/结果会在原表基础上追加 prediction 列/)).toBeVisible();

    // The upload path the browser-tool pass could not automate.
    await page.locator('.ant-upload input[type="file"]').setInputFiles(PREDICT_CSV);
    await expect(page.getByText(/已提交，正在后台预测/)).toBeVisible({ timeout: 30000 });

    // Poll through the UI itself — the panel is responsible for reaching a
    // terminal state on its own, without the user pressing refresh.
    await expect(page.getByText('已完成')).toBeVisible({ timeout: 120000 });

    // Read the job id the panel is showing, then confirm the server agrees.
    // Asserting only on the UI would pass even if the panel invented numbers.
    // `bordered` Descriptions render as a table (rowheader + cell), not as
    // .ant-descriptions-item wrappers — read the row and pull the uuid out.
    const idRow = await page.getByRole('row', { name: /任务 ID/ }).innerText();
    const jobId = (idRow.match(/[0-9a-f]{8}-[0-9a-f-]{27}/) || [])[0];
    expect(jobId, `no job id in row: ${idRow}`).toBeTruthy();

    const job = await (
      await request.get(`${API}/inference/${deploymentId}/batch-predict/${jobId}`)
    ).json();
    expect(job.status).toBe('completed');
    expect(job.processed_rows).toBe(job.input_rows);
    expect(job.processed_rows).toBeGreaterThan(0);

    // And the result is downloadable as a CSV that keeps the input columns.
    const dl = await request.get(
      `${API}/inference/${deploymentId}/batch-predict/${jobId}/download`,
    );
    expect(dl.ok()).toBeTruthy();
    expect(dl.headers()['content-type']).toContain('text/csv');
    const body = await dl.text();
    const [header, ...rows] = body.trim().split('\n');
    expect(header).toContain('prediction');
    expect(header).toContain('tenure');
    expect(rows.length).toBe(job.input_rows);
  });
});
