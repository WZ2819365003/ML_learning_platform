/**
 * buildComparisonVM — normalize a list of run/leaderboard rows into one view
 * model for the ModelComparison component.
 *
 * Both `modelingTaskApi.leaderboard` rows and `modelingTaskApi.runs().items`
 * carry `params.model_type`, `metrics`, `objective_value`, `domain_task_id`,
 * `family`, `strategy_type`. `metricKeys` is derived from the union of every
 * run's `metrics` keys plus the task's `objective_metric` (NOT task.eval_metrics
 * — the task object only exposes `objective_metric`). Missing metrics render as
 * '-' in the component.
 *
 * @param {Array} rawRows  leaderboard rows or runs().items
 * @param {{objective_metric?: string, objective_direction?: 'max'|'min'}} task
 * @returns {{rows: Array, metricKeys: string[], objective_metric: string, objective_direction: 'max'|'min'}}
 */
export function buildComparisonVM(rawRows = [], task = {}) {
  const objective_metric = task.objective_metric || 'accuracy'
  const objective_direction = task.objective_direction === 'min' ? 'min' : 'max'

  const rows = (rawRows || []).map(r => ({
    run_id: r.run_id,
    model_type: r.params?.model_type || r.model_type || '-',
    strategy_type: r.strategy_type || 'baseline',
    status: r.status || 'SUCCESS',
    metrics: r.metrics || {},
    objective_value: typeof r.objective_value === 'number' ? r.objective_value : null,
    domain_task_id: r.domain_task_id ?? null,
    family: r.family ?? null,
    is_best: false,
  }))

  // metricKeys: objective first, then the rest of the union of run.metrics keys.
  const keys = new Set([objective_metric])
  rows.forEach(r => Object.keys(r.metrics).forEach(k => keys.add(k)))
  const metricKeys = [
    objective_metric,
    ...[...keys].filter(k => k !== objective_metric).sort(),
  ]

  // is_best: single winner by objective_value + direction (ignore null values).
  const scored = rows.filter(r => r.objective_value != null)
  if (scored.length) {
    const best = scored.reduce((a, b) => (
      objective_direction === 'max'
        ? (b.objective_value > a.objective_value ? b : a)
        : (b.objective_value < a.objective_value ? b : a)
    ))
    best.is_best = true
  }

  return { rows, metricKeys, objective_metric, objective_direction }
}
