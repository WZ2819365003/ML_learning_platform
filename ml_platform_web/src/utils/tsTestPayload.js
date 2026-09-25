/**
 * 时序部署「在线测试」请求体的预填。
 *
 * 目标是打开抽屉就能直接点发送，而不是让人对着一堆空字段猜列名。
 *
 * 两级来源：
 *  1. 最近一次跑成功的时序任务——它的 dataset_id / value_column / time_column
 *     已经被后端验证过一次，照抄就一定能跑通。
 *  2. 没有历史任务时（全新环境）再去猜列名。
 *
 * 猜列名这段以前有个 bug：inferColumns 直接往外层的 timeCol / valueCol 上写，
 * 数据集被判定不合格、循环继续之后，上一轮的半个结果还留在外面。页面上就出现
 * dataset_id 为空、value_column 却是 "hour_of_day" 这种拼不出请求的组合。
 * 这里的函数一律纯返回，不写外部变量。
 */

// 目标列的名字线索，按优先级从高到低。
const VALUE_HINTS = [
  'load', 'power', 'demand', 'consumption', 'output', 'generation',
  'value', 'target', 'sales', 'price', 'volume',
  '负荷', '功率', '发电', '用电', '电量',
]

// 时间列的名字线索。
const TIME_HINTS = [
  'timestamp', 'datetime', 'date', 'time', 'hour', 'minute', 'period',
  'days_since', 'ds', '时间', '日期',
]

// 明显不是预测目标的派生特征：滞后、滑窗、三角编码、独热标志。
const DERIVED_PATTERNS = [
  /_lag_?\d*$/i, /_roll_/i, /_rolling_/i, /_sin$/i, /_cos$/i,
  /_sq$/i, /^is_/i, /_diff\d*$/i, /_pct_change$/i,
]

function lower(value) {
  return String(value ?? '').toLowerCase()
}

function isDatetimeDtype(dtype) {
  const dt = lower(dtype)
  return dt.startsWith('datetime') || dt.startsWith('timestamp') || dt === 'date'
}

function isNumericDtype(dtype) {
  const dt = lower(dtype)
  return dt.startsWith('int') || dt.startsWith('float') || dt.startsWith('uint')
    || dt.startsWith('number') || dt.startsWith('double')
}

function looksDerived(name) {
  return DERIVED_PATTERNS.some(pattern => pattern.test(name))
}

function hintRank(name, hints) {
  const key = lower(name)
  for (let i = 0; i < hints.length; i += 1) {
    if (key === hints[i]) return { rank: i, exact: true }
  }
  for (let i = 0; i < hints.length; i += 1) {
    if (key.includes(hints[i])) return { rank: i, exact: false }
  }
  return null
}

function columnEntries(preview) {
  const info = preview?.columns_info ?? {}
  const stats = preview?.statistics ?? {}
  return Object.keys(info).map(name => ({
    name,
    dtype: info[name]?.dtype,
    // 注意：unique_count 是预览抽样上算的（通常上限 100 行），不能拿它跟总行数
    // 求比例去判断 ID 列——大数据集上这个比例永远接近 0。只用来做相对排序。
    unique: Number(stats[name]?.unique_count ?? 0),
  }))
}

/**
 * 挑时间列。datetime 类型优先；否则按名字线索，同级里取抽样去重值最多的那个——
 * days_since_start 这种单调列会赢过 hour_of_day 这种周期列，画出来的轴才有意义。
 */
export function inferTimeColumn(preview) {
  const columns = columnEntries(preview)
  const datetime = columns.find(col => isDatetimeDtype(col.dtype))
  if (datetime) return datetime.name

  const candidates = columns
    .map(col => ({ col, hint: hintRank(col.name, TIME_HINTS) }))
    .filter(item => item.hint !== null)
  if (candidates.length === 0) return null

  candidates.sort((a, b) => {
    if (a.hint.exact !== b.hint.exact) return a.hint.exact ? -1 : 1
    if (a.col.unique !== b.col.unique) return b.col.unique - a.col.unique
    return a.hint.rank - b.hint.rank
  })
  return candidates[0].col.name
}

/**
 * 挑目标列。名字命中线索的优先（load 胜过 load_lag_1，因为精确匹配排在前面），
 * 都没命中就退回抽样去重值最多的数值列——常量列和标志位自然排在后面。
 */
export function inferValueColumn(preview, excludeColumn = null) {
  const numeric = columnEntries(preview).filter(col =>
    isNumericDtype(col.dtype) && col.name !== excludeColumn)
  if (numeric.length === 0) return null

  const hinted = numeric
    .map(col => ({ col, hint: hintRank(col.name, VALUE_HINTS) }))
    .filter(item => item.hint !== null && !looksDerived(item.col.name))
  if (hinted.length > 0) {
    hinted.sort((a, b) => {
      if (a.hint.exact !== b.hint.exact) return a.hint.exact ? -1 : 1
      if (a.hint.rank !== b.hint.rank) return a.hint.rank - b.hint.rank
      // 同样命中时取名字最短的：load 而不是 load_roll_mean_48
      return a.col.name.length - b.col.name.length
    })
    return hinted[0].col.name
  }

  const plain = numeric.filter(col => !looksDerived(col.name))
  const pool = plain.length > 0 ? plain : numeric
  return pool.reduce((best, col) => (col.unique > best.unique ? col : best), pool[0]).name
}

/**
 * 从数据集预览推断一份请求体。time_column 推不出来就留 null——后端签名是
 * `time_column: str | None`，只拿它画显示用的横轴，缺了不影响预测。
 */
export function payloadFromPreview(dataset, preview) {
  const timeColumn = inferTimeColumn(preview)
  const valueColumn = inferValueColumn(preview, timeColumn)
  if (!valueColumn) return null
  return {
    dataset_id: dataset?.id ?? '',
    value_column: valueColumn,
    time_column: timeColumn,
    horizon: 24,
    frequency: 'D',
  }
}

/**
 * 照抄一次跑成功的任务。这是最靠谱的预填：这组参数后端已经接受过一次。
 */
export function payloadFromTask(task) {
  if (!task?.dataset_id || !task?.value_column) return null
  return {
    dataset_id: task.dataset_id,
    value_column: task.value_column,
    time_column: task.time_column ?? null,
    horizon: Number(task.horizon) > 0 ? Number(task.horizon) : 24,
    frequency: task.frequency || 'D',
  }
}

/**
 * 从任务列表里选最近一次成功的。列表接口按创建时间倒序，这里不依赖顺序，
 * 显式按完成时间挑。
 */
export function latestSuccessfulTask(tasks) {
  const done = (tasks ?? []).filter(task =>
    String(task?.status ?? '').toUpperCase() === 'SUCCESS' && task?.dataset_id && task?.value_column)
  if (done.length === 0) return null
  return done.reduce((best, task) => {
    const at = task.finished_at ?? task.created_at ?? ''
    const bestAt = best.finished_at ?? best.created_at ?? ''
    return String(at) > String(bestAt) ? task : best
  }, done[0])
}

/** 请求体缺 dataset_id 或 value_column 就发不出去；time_column 可以为空。 */
export function missingTsFields(payload) {
  const missing = []
  if (!payload?.dataset_id) missing.push('dataset_id')
  if (!payload?.value_column) missing.push('value_column')
  return missing
}

export const TS_TEST_FALLBACK = {
  dataset_id: '',
  value_column: '',
  time_column: null,
  horizon: 24,
  frequency: 'D',
}
