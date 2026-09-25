/**
 * buildComparisonVM — normalize a list of run/leaderboard rows into one view
 * model for the ModelComparison component.
 *
 * Both `modelingTaskApi.leaderboard` rows and `modelingTaskApi.runs().items`
 * carry `params.model_type`, `metrics`, `objective_value`, `domain_task_id`,
 * `family`, `strategy_type`. Requested `eval_metrics` take precedence over a
 * numeric-metric fallback so internal CV and explanation fields do not leak
 * into the comparison table.
 *
 * @param {Array} rawRows  leaderboard rows or runs().items
 * @param {{objective_metric?: string, objective_direction?: 'max'|'min'}} task
 * @returns {{rows: Array, metricKeys: string[], objective_metric: string, objective_direction: 'max'|'min'}}
 */
export function buildComparisonVM(rawRows = [], task = {}) {
  const objective_metric = task.objective_metric || 'accuracy'
  const objective_direction = task.objective_direction === 'min' ? 'min' : 'max'
  const requestedMetrics = [
    ...(Array.isArray(task.eval_metrics) ? task.eval_metrics : []),
    ...(Array.isArray(task.config?.eval_metrics) ? task.config.eval_metrics : []),
    ...(rawRows || []).flatMap(r => Array.isArray(r.params?.eval_metrics) ? r.params.eval_metrics : []),
  ].filter(Boolean)
  const hasConfiguredMetrics = requestedMetrics.length > 0
  const configuredMetricKeys = [...new Set([objective_metric, ...requestedMetrics])]

  const metricValue = (raw, key, objectiveValue) => {
    if (key === objective_metric && typeof objectiveValue === 'number') return objectiveValue
    const metrics = raw.metrics || {}
    // selection_cv_mean_* = classic-ML CV selection score;
    // selection_val_*    = DL inner-validation selection score (B1)
    for (const candidate of [key, `selection_cv_mean_${key}`, `selection_val_${key}`, `cv_avg_${key}`]) {
      if (typeof metrics[candidate] === 'number') return metrics[candidate]
    }
    return null
  }

  const rows = (rawRows || []).map(r => {
    const objectiveValue = typeof r.selection_value === 'number'
      ? r.selection_value
      : (typeof r.objective_value === 'number' ? r.objective_value : null)
    const metrics = hasConfiguredMetrics
      ? Object.fromEntries(configuredMetricKeys.flatMap(key => {
        const value = metricValue(r, key, objectiveValue)
        return typeof value === 'number' ? [[key, value]] : []
      }))
      : { ...(r.metrics || {}) }
    if (objectiveValue != null) metrics[objective_metric] = objectiveValue
    const isSuccess = String(r.status || 'SUCCESS').toUpperCase() === 'SUCCESS'
    const hasArtifact = isSuccess && !!r.domain_task_id
    return {
      run_id: r.run_id,
      model_type: r.params?.model_type || r.model_type || '-',
      strategy_type: r.strategy_type || 'baseline',
      status: r.status || 'SUCCESS',
      metrics,
      objective_value: objectiveValue,
      selection_metric_key: r.selection_metric_key || objective_metric,
      selection_value: objectiveValue,
      final_test_metric_key: r.final_test_metric_key || null,
      final_test_value: typeof r.final_test_value === 'number' ? r.final_test_value : null,
      domain_task_id: r.domain_task_id ?? null,
      family: r.family ?? null,
      // Carried through for the expandable row. The leaderboard endpoint has
      // always returned both, but the table only ever rendered the handful of
      // configured metric columns — so the hyperparameters that actually
      // distinguish two trials of the same model were nowhere on screen.
      params: r.params || {},
      all_metrics: r.metrics || {},
      is_success: isSuccess,
      has_artifact: hasArtifact,
      can_explain: hasArtifact,
      can_download: hasArtifact,
      can_deploy: hasArtifact,
      rank: null,
      is_best: false,
    }
  })

  const metricKeys = hasConfiguredMetrics
    ? configuredMetricKeys
    : (() => {
      const keys = new Set([objective_metric])
      rows.forEach(r => Object.entries(r.metrics).forEach(([key, value]) => {
        const internal = key.startsWith('selection_cv_') || key.startsWith('selection_val_')
          || key.startsWith('final_test_') || key === 'cv_folds'
        if (!internal && typeof value === 'number') keys.add(key)
      }))
      return [objective_metric, ...[...keys].filter(k => k !== objective_metric).sort()]
    })()

  // Failed/running rows remain visible for diagnostics but are not ranked.
  const scored = rows
    .filter(r => r.is_success && Number.isFinite(r.objective_value))
    .sort((a, b) => objective_direction === 'max'
      ? b.objective_value - a.objective_value
      : a.objective_value - b.objective_value)
  scored.forEach((row, index) => { row.rank = index + 1 })
  if (scored.length) {
    scored[0].is_best = true
  }
  const orderedRows = [...rows].sort((a, b) => {
    if (a.rank == null && b.rank == null) return 0
    if (a.rank == null) return 1
    if (b.rank == null) return -1
    return a.rank - b.rank
  })

  return { rows: orderedRows, metricKeys, objective_metric, objective_direction }
}

export function buildStrategyCardVM(strategy = {}) {
  const bestRun = strategy.best_run || null
  return {
    hasBestRun: Number.isFinite(bestRun?.objective_value),
    bestRun,
    runCount: Number(strategy.run_count || 0),
    fullRunCount: Number(strategy.full_run_count || 0),
  }
}

export function buildFinalizationVM(task = {}, bestRun = null) {
  const finalEvaluation = task.final_evaluation || { state: 'OPEN', version: 1 }
  const state = String(finalEvaluation.state || 'OPEN').toUpperCase()
  const objectiveMetric = task.objective_metric || 'accuracy'
  const activeRuns = Number(task.run_stats?.running || 0)
  const hasActiveRuns = activeRuns > 0 || String(task.status).toUpperCase() === 'RUNNING'
  const finalValue = typeof finalEvaluation.final_metrics?.[`final_test_${objectiveMetric}`] === 'number'
    ? finalEvaluation.final_metrics[`final_test_${objectiveMetric}`]
    : (typeof bestRun?.final_test_value === 'number' ? bestRun.final_test_value : null)

  if (state === 'FINALIZED') {
    return {
      state, disabled: true, reason: null, actionLabel: '已确认最终模型',
      finalValue, error: null,
    }
  }
  if (state === 'EVALUATING') {
    return {
      state, disabled: true, reason: '最终评估正在执行', actionLabel: '正在确认',
      finalValue: null, error: null,
    }
  }

  let reason = null
  if (!bestRun) reason = '还没有可确认的成功 Run'
  else if (hasActiveRuns) reason = '请等待所有 Run 运行结束'
  // B1: DL winners are finalizable — only unknown families stay blocked.
  else if (bestRun.family !== 'ml' && bestRun.family !== 'dl') reason = '该模型族暂不支持最终确认'

  return {
    state,
    disabled: !!reason,
    reason,
    actionLabel: state === 'FAILED' ? '重试最终确认' : '确认最终模型',
    finalValue: null,
    error: state === 'FAILED' ? finalEvaluation.error || '最终确认失败' : null,
  }
}

export async function finalizeTaskAndRefresh(finalize, refresh) {
  try {
    return await finalize()
  } finally {
    await refresh?.()
  }
}
