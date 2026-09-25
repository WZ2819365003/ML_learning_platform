/**
 * deploySchema — builds the request/response contract shown on the 部署 step.
 *
 * All pure functions over data the app already has (a dataset preview row plus
 * the task's target column), so the examples are the caller's *real* columns
 * with real values rather than invented placeholders.
 */

/** Columns the trainer strips before fitting — never valid request fields. */
const DERIVED_TARGET_PREFIXES = ['predicted_', 'prediction_']

/**
 * Feature columns a request must supply: every dataset column except the
 * target and anything derived from it.
 */
export function requestFeatureNames(columnNames = [], targetColumn = '') {
  return columnNames.filter((name) => {
    if (!name || name === targetColumn) return false
    return !DERIVED_TARGET_PREFIXES.some((p) => name.startsWith(p))
  })
}

/**
 * A request body built from one real sample row.
 *
 * Falls back to a per-dtype placeholder for any column the sample is missing,
 * so the example is never partially blank.
 */
export function buildRequestExample({
  sampleRow = {},
  columnsInfo = {},
  columnNames = [],
  targetColumn = '',
} = {}) {
  const names = requestFeatureNames(
    columnNames.length ? columnNames : Object.keys(columnsInfo),
    targetColumn,
  )
  const row = {}
  for (const name of names) {
    const sampled = sampleRow?.[name]
    if (sampled !== undefined && sampled !== null) {
      row[name] = sampled
      continue
    }
    const dtype = String(columnsInfo?.[name]?.dtype || '')
    if (dtype.startsWith('float')) row[name] = columnsInfo[name]?.mean ?? 0.0
    else if (dtype.startsWith('int')) row[name] = Math.round(columnsInfo[name]?.mean ?? 0)
    else if (dtype.startsWith('bool')) row[name] = false
    else row[name] = '示例值'
  }
  return { rows: [row], include_probabilities: true }
}

/**
 * The response shape for this task kind.
 *
 * Mirrors InferenceJobResponse. `probabilities` is null for regression — the
 * field exists on the model either way, which is exactly the sort of thing a
 * caller writing a client needs told rather than discovered.
 */
export function buildResponseExample({ taskType = 'classification', classLabels = [] } = {}) {
  const isRegression = taskType === 'regression'
  const labels = classLabels.length ? classLabels : ['类别A', '类别B']
  return {
    job_id: 'a1b2c3d4-...',
    deployment_id: 'd5e6f7a8-...',
    status: 'completed',
    input_rows: 1,
    predictions: isRegression ? [123.456] : [labels[0]],
    probabilities: isRegression
      ? null
      : [Object.fromEntries(labels.map((l, i) => [l, i === 0 ? 0.87 : 0.13]))],
    error_message: null,
  }
}

/** Absolute predict URL for a deployment, from the page's own origin. */
export function predictUrl(deploymentId, origin) {
  const base = origin || (typeof window !== 'undefined' ? window.location.origin : '')
  return `${base}/inference/${deploymentId || '{deployment_id}'}/predict`
}

/** A runnable curl for the predict endpoint. */
export function buildCurl({ deploymentId, requestExample, origin, authToken } = {}) {
  const url = predictUrl(deploymentId, origin)
  const authLine = authToken
    ? `  -H 'Authorization: Bearer ${authToken}' \\\n`
    : `  -H 'Authorization: Bearer <你的令牌>' \\\n`
  const body = JSON.stringify(requestExample ?? { rows: [{}], include_probabilities: true })
  return `curl -X POST '${url}' \\\n  -H 'Content-Type: application/json' \\\n${authLine}  -d '${body}'`
}

/**
 * Caveats a caller cannot infer from the JSON alone.
 *
 * The encoding one matters most: training applies a LabelEncoder to object
 * columns, so a text feature is sent as its original string and encoded
 * server-side — send the integer and you get a silently wrong prediction
 * rather than an error.
 */
export function deploymentNotes({ maxBatchSize = 100, hasTextFeatures = false } = {}) {
  const notes = [
    `请求体为 \`{"rows": [...]}\`，rows 是对象数组，一次最多 ${maxBatchSize} 行。`,
    '每行必须包含全部特征列；缺列会按缺失值处理，可能得到不可靠的预测。',
    '目标列不要放进请求体 —— 它是模型要预测的东西，传了会被忽略。',
    '需要携带登录令牌：`Authorization: Bearer <token>`。',
    '大批量请走批量预测接口 `/inference/{id}/batch-predict`（文件进、文件出、异步返回任务号）。',
  ]
  if (hasTextFeatures) {
    notes.splice(2, 0,
      '文本/类别型特征请传**训练时的原始字符串**，编码由服务端完成；传编码后的数字会得到错误结果且不会报错。')
  }
  return notes
}

/**
 * Starting weights for an ensemble, derived from each member's selection score.
 *
 * Better models get more say: proportional to the score when higher is better,
 * proportional to its reciprocal when lower is better (rmse and friends). The
 * result is normalised to sum to 1 so the fused prediction stays on the scale
 * of the individual ones.
 *
 * These are a *starting point*, not a recommendation to trust blindly — equal
 * weighting is a famously hard baseline to beat, and any weighting fitted to
 * the selection scores has seen that data. Whatever comes out here still owes
 * an honest number from the sealed test set before it can be compared with a
 * single model.
 */
export function suggestWeights(members = [], direction = 'max') {
  const usable = members.filter((m) => typeof m.objective_value === 'number')
  if (usable.length === 0) {
    // Nothing to go on — split evenly rather than inventing an ordering.
    const even = members.length ? 1 / members.length : 0
    return Object.fromEntries(members.map((m) => [m.run_id, even]))
  }

  const raw = members.map((m) => {
    const v = m.objective_value
    if (typeof v !== 'number') return 0
    if (direction === 'min') return v > 0 ? 1 / v : 0
    return v > 0 ? v : 0
  })
  const total = raw.reduce((a, b) => a + b, 0)
  if (total <= 0) {
    const even = members.length ? 1 / members.length : 0
    return Object.fromEntries(members.map((m) => [m.run_id, even]))
  }
  return Object.fromEntries(members.map((m, i) => [m.run_id, raw[i] / total]))
}

/** Normalise edited weights back to sum 1, keeping their relative sizes. */
export function normaliseWeights(weights = {}) {
  const entries = Object.entries(weights).filter(([, v]) => typeof v === 'number' && v > 0)
  const total = entries.reduce((a, [, v]) => a + v, 0)
  if (total <= 0) return { ...weights }
  return Object.fromEntries(entries.map(([k, v]) => [k, v / total]))
}
