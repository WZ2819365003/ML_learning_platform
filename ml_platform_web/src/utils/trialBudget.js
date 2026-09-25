/**
 * 调参批次的试验预算：每个模型计划跑多少次、总上限会不会把后面的模型饿死。
 *
 * 与后端 tuning_service._planned_trials_per_model / _budget_starvation_detail 同一
 * 套规则——后端在派发前会 422 拦下，这里是为了在用户点提交之前就把问题摆出来。
 *
 * 「最大 Trial 数」是整个批次的总上限，按模型顺序消耗。曾经三个模型每个 20 次、
 * 上限默认 20，结果只有第一个模型跑了，另外两个一次都没训练，实验还显示已完成。
 */

function comboCount(grid) {
  return Object.values(grid || {}).reduce(
    (n, values) => n * (Array.isArray(values) ? Math.max(values.length, 1) : 1),
    1,
  )
}

/**
 * @returns {Array<{model: string, planned: number}>} 按选择顺序；展开时会被跳过的模型不出现
 */
export function plannedTrialsPerModel(strategy, selectedModels, tuningSpaces, modelParams, nTrialsPerModel) {
  const out = []
  for (const model of selectedModels || []) {
    const spec = tuningSpaces?.[model] || {}
    if (spec.family === 'dl') continue
    if (strategy === 'grid_search') {
      // 后端：用户网格非空就用用户的，否则用注册表 grid_values（不合并）
      const userGrid = modelParams?.grid?.[model]
      const grid = userGrid && Object.keys(userGrid).length ? userGrid : spec.grid_values
      if (!grid || !Object.keys(grid).length) continue
      out.push({ model, planned: comboCount(grid) })
    } else if (strategy === 'bayesian_search') {
      const userDist = modelParams?.distribution?.[model]
      const dist = userDist && Object.keys(userDist).length ? userDist : spec.distribution
      if (!dist || !Object.keys(dist).length) continue
      out.push({ model, planned: Number(nTrialsPerModel) > 0 ? Number(nTrialsPerModel) : 10 })
    }
  }
  return out
}

/**
 * 按模型顺序模拟消耗总上限。
 * @returns {{ total: number, starved: Array<{model, planned, got}> }}
 *   starved 为空表示没问题；只选一个模型时截断属于正常的安全阀，不算饿死。
 */
export function budgetStarvation(planned, maxTrials) {
  const total = planned.reduce((n, p) => n + p.planned, 0)
  const cap = Number(maxTrials)
  if (!cap || planned.length < 2 || cap >= total) return { total, starved: [] }
  let remaining = cap
  const starved = []
  for (const p of planned) {
    const got = Math.min(p.planned, remaining)
    remaining -= got
    if (got < p.planned) starved.push({ ...p, got })
  }
  return { total, starved }
}
