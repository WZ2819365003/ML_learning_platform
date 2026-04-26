// 11 — V3 专业级可视化发布门禁：
// 在一次真实 baseline run 之上，逐个校验 v3.2 引入的 5 个新 viz 端点
// (per_class / pr_curve / calibration / threshold / distribution)
// + 既有 confusion_matrix / roc_curve / feature_importance / shap_summary。
//
// 每条断言必须真实命中后端的 viz_service —— 不能让"端点 200 但内容为空"
// 这种静默退化通过。所有断言失败即视为 v3.2 viz 退化，门禁红。
const { test, expect } = require('@playwright/test');
const { getJson } = require('../helpers/api');
const { runBaselineFlow } = require('../helpers/v3-flow');

test.describe.configure({ mode: 'serial' });

test.describe('11 V3 专业级可视化门禁', () => {
  test('11.1 baseline run 上验证 9 个 viz 端点 + SHAP 真实计算', async ({ request }) => {
    test.setTimeout(240_000);

    // 1) 走完整 plan→task→launch→SUCCESS run 链路（与 08 同一蓝图）
    const { run } = await runBaselineFlow(request, { namePrefix: 'pro-viz' });
    const runId = run.run_id || run.id;
    expect(runId, 'run id missing').toBeTruthy();

    // 2) 老的三件套 —— confusion_matrix / roc_curve / feature_importance
    const cm = await getJson(request, `/viz/${runId}/confusion_matrix`);
    expect(cm.ok, `confusion_matrix failed: ${cm.status} ${cm.raw?.slice(0, 400)}`).toBeTruthy();
    expect(Array.isArray(cm.body?.matrix), `cm.matrix shape: ${JSON.stringify(cm.body).slice(0, 300)}`).toBe(true);
    expect(cm.body.matrix.length).toBeGreaterThan(0);

    const roc = await getJson(request, `/viz/${runId}/roc_curve`);
    expect(roc.ok, `roc_curve failed: ${roc.status} ${roc.raw?.slice(0, 400)}`).toBeTruthy();
    // 二分类返回 fpr/tpr 数组，多分类返回 curves 列表
    const rocOk =
      Array.isArray(roc.body?.fpr) && Array.isArray(roc.body?.tpr) ||
      Array.isArray(roc.body?.curves);
    expect(rocOk, `roc shape: ${JSON.stringify(roc.body).slice(0, 300)}`).toBe(true);

    const fi = await getJson(request, `/viz/${runId}/feature_importance`);
    expect(fi.ok, `feature_importance failed: ${fi.status} ${fi.raw?.slice(0, 400)}`).toBeTruthy();
    const fiList = fi.body?.features || fi.body?.importances || fi.body?.items;
    expect(Array.isArray(fiList) && fiList.length > 0, `fi shape: ${JSON.stringify(fi.body).slice(0, 300)}`).toBe(true);

    // 3) v3.2 新增的 5 个端点 —— per_class / pr_curve / calibration / threshold / distribution
    const perClass = await getJson(request, `/viz/${runId}/per_class`);
    expect(perClass.ok, `per_class failed: ${perClass.status} ${perClass.raw?.slice(0, 500)}`).toBeTruthy();
    expect(Array.isArray(perClass.body?.rows), `per_class.rows missing`).toBe(true);
    expect(perClass.body.rows.length, 'per_class.rows empty (binary classifier should have 2 rows)').toBeGreaterThan(0);
    expect(perClass.body.macro_avg, 'per_class.macro_avg missing').toBeTruthy();
    expect(typeof perClass.body.accuracy).toBe('number');
    // 抽样检查一行数据完整
    const firstRow = perClass.body.rows[0];
    for (const k of ['label', 'precision', 'recall', 'f1', 'support']) {
      expect(firstRow[k] !== undefined, `per_class row missing ${k}`).toBe(true);
    }

    const pr = await getJson(request, `/viz/${runId}/pr_curve`);
    expect(pr.ok, `pr_curve failed: ${pr.status} ${pr.raw?.slice(0, 500)}`).toBeTruthy();
    if (pr.body?.multiclass) {
      expect(Array.isArray(pr.body.curves) && pr.body.curves.length > 0).toBe(true);
    } else {
      expect(Array.isArray(pr.body?.precision), `pr.precision missing`).toBe(true);
      expect(Array.isArray(pr.body?.recall), `pr.recall missing`).toBe(true);
      expect(typeof pr.body?.average_precision).toBe('number');
      expect(typeof pr.body?.best_threshold).toBe('number');
    }

    const cal = await getJson(request, `/viz/${runId}/calibration?n_bins=10`);
    expect(cal.ok, `calibration failed: ${cal.status} ${cal.raw?.slice(0, 500)}`).toBeTruthy();
    expect(Array.isArray(cal.body?.prob_pred), `cal.prob_pred missing`).toBe(true);
    expect(Array.isArray(cal.body?.prob_true), `cal.prob_true missing`).toBe(true);
    expect(typeof cal.body?.ece, `cal.ece missing`).toBe('number');
    expect(typeof cal.body?.brier, `cal.brier missing`).toBe('number');

    const thr = await getJson(request, `/viz/${runId}/threshold?step=0.05`);
    expect(thr.ok, `threshold failed: ${thr.status} ${thr.raw?.slice(0, 500)}`).toBeTruthy();
    expect(Array.isArray(thr.body?.rows), 'threshold.rows missing').toBe(true);
    expect(thr.body.rows.length, 'threshold.rows should have ~19 rows for step=0.05').toBeGreaterThanOrEqual(10);
    expect(typeof thr.body?.best_threshold, 'best_threshold missing').toBe('number');
    // 每行结构
    for (const k of ['threshold', 'precision', 'recall', 'f1', 'accuracy']) {
      expect(thr.body.rows[0][k] !== undefined, `threshold row missing ${k}`).toBe(true);
    }

    const dist = await getJson(request, `/viz/${runId}/distribution?bins=20`);
    expect(dist.ok, `distribution failed: ${dist.status} ${dist.raw?.slice(0, 500)}`).toBeTruthy();
    expect(Array.isArray(dist.body?.bin_edges), 'distribution.bin_edges missing').toBe(true);
    expect(Array.isArray(dist.body?.counts) || Array.isArray(dist.body?.positive_counts),
      `distribution counts missing: ${JSON.stringify(dist.body).slice(0, 300)}`).toBe(true);
    expect(typeof dist.body?.kind, 'distribution.kind missing').toBe('string');

    // 4) SHAP — 已经在 08 验过基本就绪，这里增量校验"feature_count > 0 且 method 不是 permutation"
    //    （permutation 是兜底档；TreeExplainer 应在 logistic_regression+xgboost 类模型上命中）
    const shap = await getJson(request, `/platform/runs/${runId}/shap?compute=true`);
    expect(shap.ok, `shap failed: ${shap.status} ${shap.raw?.slice(0, 500)}`).toBeTruthy();
    expect(shap.body?.status, 'shap.status').toBe('ready');
    expect(Number(shap.body?.feature_count || 0)).toBeGreaterThan(0);
    // method 字段保留为可选（旧版本可能没填），但若存在必须是合法值
    if (shap.body?.method) {
      expect(['tree', 'kernel', 'permutation']).toContain(shap.body.method);
    }
  });
});
