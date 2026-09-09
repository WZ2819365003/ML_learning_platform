/**
 * reportCharts — the one place report figures get their look.
 *
 * The backend emits semantic chart specs ({kind, title, caption, series,
 * tooltip_fields, …}); `renderReportChart(spec)` maps a spec to an ECharts
 * option. Colours, font sizes, margins, tooltip format and height live in
 * REPORT_CHART_THEME below and nowhere else, so a styling fix is a one-file
 * change rather than a per-chart patch.
 *
 * Pure functions only: no React, no echarts import. The component that mounts
 * the option (ReportDocument.jsx) registers the ECharts modules these options
 * need (custom series for error bars, markArea for shaded ranges).
 *
 * Legacy archives carry a ready-made ECharts option (`spec.option`); those pass
 * through untouched so old reports keep rendering.
 */

export const REPORT_CHART_THEME = Object.freeze({
  colors: Object.freeze({
    primary: '#2563eb',
    primaryLight: '#93c5fd',
    muted: '#94a3b8',
    mutedLight: '#cbd5e1',
    accent: '#dc2626',
    ink: '#0f172a',
    text: '#334155',
    subtle: '#64748b',
    grid: 'rgba(15, 23, 42, 0.08)',
    axis: 'rgba(15, 23, 42, 0.28)',
    shade: 'rgba(37, 99, 235, 0.08)',
  }),
  // Segment colours for the stacked bar cycle through these.
  palette: Object.freeze(['#2563eb', '#60a5fa', '#94a3b8', '#cbd5e1', '#1d4ed8', '#bfdbfe']),
  fontSize: 12,
  titleFontSize: 13,
  fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif',
  height: 300,
  // A single stacked bar at 300px is mostly empty; the exception is documented
  // here rather than sprinkled on the component.
  heights: Object.freeze({ stacked: 200 }),
  // `top` leaves a line for marker labels, which sit upright above the plot.
  grid: Object.freeze({ top: 28, right: 32, bottom: 32, legendTop: 44, axisNameBottom: 48 }),
  // Longest y label we will lay out on one line; anything longer wraps.
  maxLabelWidth: 200,
  labelMargin: 8,
  labelGutter: 12,
  tooltipMaxItems: 20,
  defaultDecimals: 4,
})

const T = REPORT_CHART_THEME
const CJK = /[\u2E80-\u9FFF\uF900-\uFAFF\uFF00-\uFFEF]/

/** Width in pixels a label needs; CJK glyphs are square, Latin ~0.58em. */
export function estimateTextWidth(text, fontSize = T.fontSize) {
  let width = 0
  for (const char of String(text ?? '')) {
    width += CJK.test(char) ? fontSize : fontSize * 0.58
  }
  return Math.ceil(width)
}

/** Left grid margin that fits the longest category label without clipping. */
export function categoryAxisLeft(categories = []) {
  const longest = categories.reduce(
    (max, label) => Math.max(max, estimateTextWidth(label)), 0,
  )
  return Math.min(longest, T.maxLabelWidth) + T.labelMargin + T.labelGutter
}

/** Height for a spec's kind — the theme's uniform height with documented exceptions. */
export function chartHeight(spec) {
  if (spec?.option && Number.isFinite(spec.height)) return spec.height
  return T.heights[spec?.kind] ?? T.height
}

/**
 * Format a value for the tooltip.
 *
 * `format` is optional on tooltip_fields: 'int', 'pct' (fraction → 12.3%),
 * 'percent' (already a percentage → 12.3%), '.Nf' / 'fixed:N' / 'N' (N
 * decimals), 'text'. Unformatted numbers get 4 decimals; integers stay whole.
 */
export function formatValue(value, format) {
  if (value === null || value === undefined || value === '') return '—'
  if (Array.isArray(value)) return formatList(value)
  if (typeof value !== 'number') {
    const asNumber = Number(value)
    if (format && format !== 'text' && Number.isFinite(asNumber) && String(value).trim() !== '') {
      return formatValue(asNumber, format)
    }
    // Tooltips are HTML; names come from user data, so text is escaped here.
    return escapeHtml(String(value))
  }
  if (!Number.isFinite(value)) return '—'
  const decimals = decimalsOf(format)
  if (format === 'int') return String(Math.round(value))
  if (format === 'pct') return formatPercent(value * 100)
  if (format === 'percent') return formatPercent(value)
  if (decimals !== null) return value.toFixed(decimals)
  if (Number.isInteger(value)) return String(value)
  return value.toFixed(T.defaultDecimals)
}

function decimalsOf(format) {
  if (typeof format !== 'string') return null
  const dot = /^\.(\d)f$/.exec(format)
  if (dot) return Number(dot[1])
  const fixed = /^fixed:(\d)$/.exec(format)
  if (fixed) return Number(fixed[1])
  if (/^\d$/.test(format)) return Number(format)
  const zeros = /^0\.(0+)$/.exec(format)
  if (zeros) return zeros[1].length
  return null
}

/** One decimal, whole numbers bare; a share under 0.05% keeps two decimals rather than showing 0%. */
function formatPercent(value) {
  const text = value.toFixed(1)
  if (text === '0.0' && value > 0) return `${value.toFixed(2)}%`
  return `${text.replace(/\.0$/, '')}%`
}

/** Lists (e.g. the column names of a feature group) wrap and cap at 20. */
function formatList(items) {
  const shown = items.slice(0, T.tooltipMaxItems).map((item) => escapeHtml(String(item)))
  if (items.length > T.tooltipMaxItems) shown.push(`等 ${items.length} 列`)
  return shown.join('<br/>')
}

function escapeHtml(text) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
}

/** One tooltip line per field: `label: value`. */
export function formatTooltipRows(fields = [], row = {}) {
  const lines = fields
    .filter((field) => field && field.key)
    .map((field) => {
      const value = formatValue(row?.[field.key], field.format)
      const label = escapeHtml(String(field.label ?? field.key))
      return `<div class="ai-report-tip-row"><span class="ai-report-tip-label">${label}</span>: ${value}</div>`
    })
  return lines.join('')
}

function tooltipBase(trigger = 'item') {
  return {
    trigger,
    confine: true,
    appendToBody: false,
    backgroundColor: 'rgba(255, 255, 255, 0.98)',
    borderColor: T.colors.grid,
    borderWidth: 1,
    padding: [8, 12],
    textStyle: { color: T.colors.ink, fontSize: T.fontSize, fontFamily: T.fontFamily },
    extraCssText: 'box-shadow: 0 6px 18px rgba(15,23,42,0.12); max-width: 360px; white-space: normal;',
  }
}

/**
 * A tooltip whose body is the spec's tooltip_fields read off the hovered row.
 * `resolveRow(params, first)` returns `{ row, heading? }`, a ready HTML string,
 * or null to show nothing.
 */
function fieldTooltip(spec, resolveRow, trigger = 'item') {
  const fields = spec.tooltip_fields?.length ? spec.tooltip_fields : DEFAULT_FIELDS[spec.kind] || []
  return {
    ...tooltipBase(trigger),
    formatter: (params) => {
      const first = Array.isArray(params) ? params[0] : params
      const resolved = resolveRow(params, first)
      if (!resolved) return ''
      if (typeof resolved === 'string') return resolved
      const body = formatTooltipRows(fields, resolved.row)
      return resolved.heading
        ? `<div class="ai-report-tip-head">${escapeHtml(String(resolved.heading))}</div>${body}`
        : body
    },
  }
}

const DEFAULT_FIELDS = {
  hbar: [{ key: 'category', label: '类别' }, { key: 'value', label: '数值' }],
  dots: [{ key: 'fold', label: '折', format: 'int' }, { key: 'rmse', label: 'RMSE' }, { key: 'mae', label: 'MAE' }, { key: 'r2', label: 'R²' }],
  hist: [{ key: 'range', label: '区间' }, { key: 'count', label: '样本数', format: 'int' }, { key: 'pct', label: '占比', format: 'percent' }],
  stacked: [{ key: 'name', label: '类别' }, { key: 'count', label: '列数', format: 'int' }, { key: 'items', label: '列名' }],
  lines: [{ key: 'epoch', label: '轮', format: 'int' }],
  scatter_pair: [{ key: 'x', label: '位置' }, { key: 'actual', label: '实际' }, { key: 'predicted', label: '预测' }],
}

function rowOf(param) {
  const row = param?.data?.row
  return row ? { row } : null
}

const axisText = () => ({ color: T.colors.text, fontSize: T.fontSize, fontFamily: T.fontFamily })
const axisLine = () => ({ lineStyle: { color: T.colors.axis } })
const splitLine = () => ({ lineStyle: { color: T.colors.grid } })

function valueAxis(extra = {}) {
  return {
    type: 'value',
    axisLabel: { ...axisText(), hideOverlap: true },
    axisLine: { show: false },
    axisTick: { show: false },
    splitLine: splitLine(),
    nameTextStyle: { ...axisText(), color: T.colors.subtle },
    nameLocation: 'middle',
    nameGap: 28,
    ...extra,
  }
}

function categoryAxis(categories, extra = {}) {
  const longest = categories.reduce((max, c) => Math.max(max, estimateTextWidth(c)), 0)
  const wrap = longest > T.maxLabelWidth
  return {
    type: 'category',
    data: categories,
    axisLabel: {
      ...axisText(),
      margin: T.labelMargin,
      interval: 0,
      ...(wrap ? { width: T.maxLabelWidth, overflow: 'break' } : {}),
    },
    axisLine: axisLine(),
    axisTick: { show: false },
    ...extra,
  }
}

function legend(names, top = 0) {
  return {
    show: names.length > 1,
    top,
    left: 0,
    itemWidth: 12,
    itemHeight: 8,
    icon: 'roundRect',
    textStyle: { ...axisText(), fontSize: T.fontSize },
    data: names,
  }
}

function roleColor(role) {
  if (role === 'accent') return T.colors.accent
  if (role === 'muted') return T.colors.muted
  return T.colors.primary
}

/**
 * A dashed vertical line at `item[valueKey]` labelled `item[labelKey]`.
 *
 * The label sits upright above the plot (ECharts would otherwise rotate it
 * along the line). "Above" is the line's end on a normal y axis and its start
 * on an inverted one — the hbar and dots charts invert theirs so the first
 * category reads at the top — hence `invertedAxis`.
 */
function verticalMarkLine(items, valueKey = 'value', labelKey = 'label', invertedAxis = false) {
  return {
    symbol: 'none',
    silent: true,
    animation: false,
    lineStyle: { color: T.colors.accent, type: 'dashed', width: 1.2 },
    label: {
      show: true,
      formatter: '{b}',
      position: invertedAxis ? 'start' : 'end',
      rotate: 0,
      distance: 4,
      color: T.colors.accent,
      fontSize: T.fontSize,
      fontFamily: T.fontFamily,
    },
    data: (items || []).map((item) => ({ xAxis: item[valueKey], name: item[labelKey] || '' })),
  }
}

// ── hbar ─────────────────────────────────────────────────────────────────────

function renderHbar(spec) {
  const categories = spec.categories || []
  const seriesSpecs = spec.series || []
  const rows = spec.rows || []
  const hasLegend = seriesSpecs.length > 1
  const left = categoryAxisLeft(categories)
  const top = hasLegend ? T.grid.legendTop : T.grid.top
  const bottom = spec.unit ? T.grid.axisNameBottom : T.grid.bottom
  const barLayout = { barGap: '10%', barCategoryGap: '35%' }

  const bars = seriesSpecs.map((s, seriesIndex) => ({
    id: `bar-${seriesIndex}`,
    name: s.name,
    type: 'bar',
    ...barLayout,
    itemStyle: { color: roleColor(s.color_role), borderRadius: [0, 2, 2, 0] },
    emphasis: { focus: 'none' },
    data: categories.map((category, i) => ({
      value: s.values?.[i] ?? null,
      row: rows[i] || { category, value: s.values?.[i] ?? null },
    })),
    ...(seriesIndex === 0 && spec.reference_lines?.length
      ? { markLine: verticalMarkLine(spec.reference_lines, 'value', 'label', true) }
      : {}),
  }))

  const errorBars = seriesSpecs
    .map((s, seriesIndex) => ({ s, seriesIndex }))
    .filter(({ s }) => Array.isArray(s.error) && s.error.some((e) => Number.isFinite(e)))
    .map(({ s, seriesIndex }) => ({
      id: `error-${seriesIndex}`,
      name: `${s.name} 误差`,
      type: 'custom',
      silent: true,
      z: 3,
      tooltip: { show: false },
      encode: { x: [0, 1], y: 2 },
      itemStyle: { borderWidth: 1.4, color: T.colors.ink },
      data: categories.map((_, i) => {
        const value = s.values?.[i]
        const error = s.error?.[i]
        if (!Number.isFinite(value) || !Number.isFinite(error)) return [null, null, i]
        return [value - error, value + error, i]
      }),
      renderItem: (params, api) => {
        const low = api.value(0)
        const high = api.value(1)
        if (!Number.isFinite(low) || !Number.isFinite(high)) return null
        const layout = api.barLayout({ ...barLayout, count: seriesSpecs.length })
        const yCenter = api.coord([low, api.value(2)])[1] + (layout[seriesIndex]?.offsetCenter || 0)
        const xLow = api.coord([low, 0])[0]
        const xHigh = api.coord([high, 0])[0]
        const cap = Math.max(3, Math.min(6, (layout[seriesIndex]?.width || 12) / 3))
        const style = api.style({ stroke: T.colors.ink, fill: 'none', lineWidth: 1.4 })
        return {
          type: 'group',
          children: [
            { type: 'line', shape: { x1: xLow, y1: yCenter, x2: xHigh, y2: yCenter }, style },
            { type: 'line', shape: { x1: xLow, y1: yCenter - cap, x2: xLow, y2: yCenter + cap }, style },
            { type: 'line', shape: { x1: xHigh, y1: yCenter - cap, x2: xHigh, y2: yCenter + cap }, style },
          ],
        }
      },
    }))

  return {
    animationDuration: 300,
    textStyle: { fontFamily: T.fontFamily },
    legend: legend(seriesSpecs.map((s) => s.name), 0),
    grid: { left, right: T.grid.right, top, bottom, containLabel: false },
    xAxis: valueAxis({ name: spec.unit || '', scale: false }),
    yAxis: categoryAxis(categories, { inverse: true }),
    tooltip: fieldTooltip(spec, (_params, first) => rowOf(first)),
    series: [...bars, ...errorBars],
  }
}

// ── dots ─────────────────────────────────────────────────────────────────────

function renderDots(spec) {
  const categories = spec.categories || []
  const points = spec.points || []
  const left = categoryAxisLeft(categories)
  const bottom = spec.unit ? T.grid.axisNameBottom : T.grid.bottom

  const dotData = []
  const meanData = []
  points.forEach((point) => {
    const ci = categories.indexOf(point.category)
    if (ci < 0) return
    const values = (point.values || []).filter((v) => Number.isFinite(v))
    values.forEach((value, k) => {
      dotData.push({ value: [value, ci], row: point.rows?.[k] || { fold: k + 1, value } })
    })
    if (spec.mean_marker && values.length) {
      const mean = values.reduce((sum, v) => sum + v, 0) / values.length
      meanData.push({ value: [mean, ci], mean, category: point.category })
    }
  })

  return {
    animationDuration: 300,
    textStyle: { fontFamily: T.fontFamily },
    grid: { left, right: T.grid.right, top: T.grid.top, bottom, containLabel: false },
    xAxis: valueAxis({ name: spec.unit || '', scale: true }),
    yAxis: categoryAxis(categories, { inverse: true, boundaryGap: true }),
    tooltip: fieldTooltip(spec, (_params, first) => {
      if (first?.seriesId === 'mean') {
        return `<div class="ai-report-tip-head">${escapeHtml(String(first.data.category))}</div>`
          + `<div class="ai-report-tip-row"><span class="ai-report-tip-label">均值</span>: ${formatValue(first.data.mean)}</div>`
      }
      return rowOf(first)
    }),
    series: [
      {
        id: 'dots',
        name: spec.title,
        type: 'scatter',
        symbolSize: 10,
        itemStyle: { color: T.colors.primary, opacity: 0.75 },
        emphasis: { focus: 'none', itemStyle: { opacity: 1 } },
        data: dotData,
      },
      {
        id: 'mean',
        name: '均值',
        type: 'scatter',
        symbol: 'rect',
        symbolSize: [3, 22],
        itemStyle: { color: T.colors.ink },
        emphasis: { focus: 'none' },
        z: 3,
        data: meanData,
      },
    ],
  }
}

// ── hist ─────────────────────────────────────────────────────────────────────

function binRow(bin) {
  const from = formatValue(bin.from, Number.isInteger(bin.from) && Number.isInteger(bin.to) ? 'int' : '.1f')
  const to = formatValue(bin.to, Number.isInteger(bin.from) && Number.isInteger(bin.to) ? 'int' : '.1f')
  return { ...bin, range: `${from} – ${to}` }
}

function histSeries(id, bins, color) {
  return {
    id,
    type: 'bar',
    barCategoryGap: '6%',
    itemStyle: { color, borderRadius: [2, 2, 0, 0] },
    emphasis: { focus: 'none' },
    data: bins.map((bin) => ({ value: [(bin.from + bin.to) / 2, bin.count], row: binRow(bin) })),
  }
}

function histAxes(bins, unit) {
  const from = bins.length ? bins[0].from : undefined
  const to = bins.length ? bins[bins.length - 1].to : undefined
  return {
    xAxis: valueAxis({ name: unit || '', min: from, max: to, splitLine: { show: false }, axisLine: axisLine() }),
    yAxis: valueAxis({ name: '样本数', nameGap: 44, minInterval: 1 }),
  }
}

function renderHist(spec) {
  const bins = spec.bins || []
  const series = histSeries('hist', bins, T.colors.primary)
  if (spec.markers?.length) series.markLine = verticalMarkLine(spec.markers)
  return {
    animationDuration: 300,
    textStyle: { fontFamily: T.fontFamily },
    grid: {
      left: 64, right: T.grid.right, top: T.grid.top + 8,
      bottom: spec.unit ? T.grid.axisNameBottom : T.grid.bottom, containLabel: false,
    },
    ...histAxes(bins, spec.unit),
    tooltip: fieldTooltip(spec, (_params, first) => rowOf(first)),
    series: [series],
  }
}

// ── stacked ──────────────────────────────────────────────────────────────────

function renderStacked(spec) {
  const segments = spec.segments || []
  const total = segments.reduce((sum, s) => sum + (Number(s.count) || 0), 0)
  return {
    animationDuration: 300,
    textStyle: { fontFamily: T.fontFamily },
    legend: {
      ...legend(segments.map((s) => s.name)),
      show: segments.length > 0,
      top: 'auto',
      bottom: 0,
      formatter: (name) => {
        const seg = segments.find((s) => s.name === name)
        return seg ? `${name} ${seg.count}` : name
      },
    },
    grid: { left: 0, right: 0, top: 24, bottom: 44, containLabel: false },
    xAxis: { type: 'value', show: false, max: total || undefined },
    yAxis: { type: 'category', data: [''], show: false },
    tooltip: fieldTooltip(spec, (_params, first) => rowOf(first)),
    series: segments.map((seg, i) => ({
      id: `seg-${i}`,
      name: seg.name,
      type: 'bar',
      stack: 'total',
      barWidth: 40,
      itemStyle: { color: T.palette[i % T.palette.length] },
      emphasis: { focus: 'none' },
      label: {
        show: true,
        position: 'inside',
        color: i % T.palette.length === 3 || i % T.palette.length === 5 ? T.colors.ink : '#ffffff',
        fontSize: T.fontSize,
        fontFamily: T.fontFamily,
        formatter: () => `${seg.name} ${seg.count}`,
        overflow: 'truncate',
      },
      data: [{ value: seg.count, row: { ...seg, pct: total ? seg.count / total : 0 } }],
    })),
  }
}

// ── lines ────────────────────────────────────────────────────────────────────

function renderLines(spec) {
  const x = spec.x || []
  const seriesSpecs = spec.series || []
  const rows = spec.rows || []
  const colors = [T.colors.primary, T.colors.accent, T.colors.muted, T.colors.ink]
  const hasLegend = seriesSpecs.length > 1

  const series = seriesSpecs.map((s, i) => ({
    id: `line-${i}`,
    name: s.name,
    type: 'line',
    showSymbol: false,
    symbol: 'circle',
    symbolSize: 6,
    smooth: false,
    lineStyle: { width: 2, color: colors[i % colors.length] },
    itemStyle: { color: colors[i % colors.length] },
    emphasis: { focus: 'none' },
    data: x.map((xi, k) => [xi, positiveOrNull(s.values?.[k], spec.y_log)]),
  }))
  if (series.length) {
    if (spec.markers?.length) series[0].markLine = verticalMarkLine(spec.markers, 'x', 'label')
    if (spec.shade && Number.isFinite(spec.shade.from) && Number.isFinite(spec.shade.to)) {
      series[0].markArea = {
        silent: true,
        itemStyle: { color: T.colors.shade },
        label: {
          show: Boolean(spec.shade.label),
          position: 'insideTop',
          color: T.colors.subtle,
          fontSize: T.fontSize,
          fontFamily: T.fontFamily,
        },
        data: [[{ xAxis: spec.shade.from, name: spec.shade.label || '' }, { xAxis: spec.shade.to }]],
      }
    }
  }

  const fields = spec.tooltip_fields?.length ? spec.tooltip_fields : null
  return {
    animationDuration: 300,
    textStyle: { fontFamily: T.fontFamily },
    legend: legend(seriesSpecs.map((s) => s.name), 0),
    grid: {
      left: 64, right: T.grid.right, top: hasLegend ? T.grid.legendTop : T.grid.top,
      bottom: T.grid.axisNameBottom, containLabel: false,
    },
    xAxis: valueAxis({
      name: spec.x_label || 'epoch', min: x.length ? x[0] : undefined, max: x.length ? x[x.length - 1] : undefined,
      splitLine: { show: false }, axisLine: axisLine(), minInterval: 1,
    }),
    yAxis: valueAxis({ type: spec.y_log ? 'log' : 'value', name: spec.unit || '', nameGap: 44, scale: true }),
    tooltip: fieldTooltip(
      { ...spec, tooltip_fields: fields || [] },
      (params, first) => {
        const index = first?.dataIndex ?? -1
        const row = rows[index]
        if (fields && row) return { row }
        // No per-epoch rows: show the series values at this x.
        const list = Array.isArray(params) ? params : [params]
        const lines = list.map((p) => (
          `<div class="ai-report-tip-row"><span class="ai-report-tip-label">${escapeHtml(String(p.seriesName))}</span>: ${formatValue(p.value?.[1])}</div>`
        ))
        return `<div class="ai-report-tip-head">${escapeHtml(`${spec.x_label || 'epoch'} ${first?.value?.[0] ?? ''}`)}</div>${lines.join('')}`
      },
      'axis',
    ),
    axisPointer: { lineStyle: { color: T.colors.axis } },
    series,
  }
}

function positiveOrNull(value, log) {
  if (!Number.isFinite(value)) return null
  if (log && value <= 0) return null
  return value
}

// ── scatter_pair ─────────────────────────────────────────────────────────────

function renderScatterPair(spec) {
  const pair = spec.pair || { x: [], actual: [], predicted: [] }
  const bins = spec.residual_bins || []
  const stats = spec.stats || {}
  const x = pair.x || []
  const rows = x.map((xi, i) => ({
    x: xi,
    actual: pair.actual?.[i] ?? null,
    predicted: pair.predicted?.[i] ?? null,
    residual: Number.isFinite(pair.actual?.[i]) && Number.isFinite(pair.predicted?.[i])
      ? pair.actual[i] - pair.predicted[i] : null,
  }))
  const split = 56
  const leftGrid = { left: 56, right: `${100 - split + 4}%`, top: 44, bottom: 52 }
  const rightGrid = { left: `${split + 2}%`, right: 24, top: 44, bottom: 52 }
  const statsText = ['rmse', 'mae'].filter((k) => Number.isFinite(stats[k]))
    .map((k) => `${k.toUpperCase()} ${formatValue(stats[k], '.1f')}`)
  if (Number.isFinite(stats.ratio)) statsText.push(`RMSE/MAE ${formatValue(stats.ratio, '.2f')}`)

  const histAxis = histAxes(bins, '残差')
  const leftFields = spec.tooltip_fields?.length ? spec.tooltip_fields : DEFAULT_FIELDS.scatter_pair
  return {
    animationDuration: 300,
    textStyle: { fontFamily: T.fontFamily },
    title: [
      {
        text: '实际 vs 预测', subtext: statsText.join(' · '), left: leftGrid.left, top: 0,
        textStyle: { fontSize: T.titleFontSize, color: T.colors.ink, fontWeight: 600, fontFamily: T.fontFamily },
        subtextStyle: { fontSize: T.fontSize, color: T.colors.subtle, fontFamily: T.fontFamily },
        itemGap: 2,
      },
      {
        text: '残差分布', left: rightGrid.left, top: 0,
        textStyle: { fontSize: T.titleFontSize, color: T.colors.ink, fontWeight: 600, fontFamily: T.fontFamily },
      },
    ],
    legend: { ...legend(['实际', '预测'], 0), left: 'auto', right: `${100 - split + 4}%`, top: 4 },
    grid: [
      { ...leftGrid, containLabel: false },
      { ...rightGrid, containLabel: false },
    ],
    xAxis: [
      {
        ...categoryAxis(x.map(String), { gridIndex: 0, boundaryGap: false }),
        axisLabel: { ...axisText(), hideOverlap: true, interval: 'auto' },
        name: spec.x_label || '', nameLocation: 'middle', nameGap: 28,
        nameTextStyle: { ...axisText(), color: T.colors.subtle },
      },
      { ...histAxis.xAxis, gridIndex: 1 },
    ],
    yAxis: [
      valueAxis({ gridIndex: 0, name: spec.unit || '', nameGap: 40, scale: true }),
      { ...histAxis.yAxis, gridIndex: 1 },
    ],
    tooltip: {
      ...tooltipBase('axis'),
      formatter: (params) => {
        const list = Array.isArray(params) ? params : [params]
        const first = list[0]
        if (!first) return ''
        if (first.seriesId === 'residual') {
          return formatTooltipRows(DEFAULT_FIELDS.hist.filter((f) => f.key !== 'pct'), first.data?.row || {})
        }
        const row = rows[first.dataIndex]
        return row ? formatTooltipRows(leftFields, row) : ''
      },
    },
    axisPointer: { link: [], lineStyle: { color: T.colors.axis } },
    series: [
      {
        id: 'actual', name: '实际', type: 'line', xAxisIndex: 0, yAxisIndex: 0,
        showSymbol: false, lineStyle: { width: 1.6, color: T.colors.muted }, itemStyle: { color: T.colors.muted },
        emphasis: { focus: 'none' }, data: x.map((_, i) => pair.actual?.[i] ?? null),
      },
      {
        id: 'predicted', name: '预测', type: 'line', xAxisIndex: 0, yAxisIndex: 0,
        showSymbol: false, lineStyle: { width: 1.6, color: T.colors.primary }, itemStyle: { color: T.colors.primary },
        emphasis: { focus: 'none' }, data: x.map((_, i) => pair.predicted?.[i] ?? null),
      },
      { ...histSeries('residual', bins, T.colors.primary), xAxisIndex: 1, yAxisIndex: 1 },
    ],
  }
}

// ── dispatch ─────────────────────────────────────────────────────────────────

const RENDERERS = {
  hbar: renderHbar,
  dots: renderDots,
  hist: renderHist,
  stacked: renderStacked,
  lines: renderLines,
  scatter_pair: renderScatterPair,
}

export const REPORT_CHART_KINDS = Object.freeze(Object.keys(RENDERERS))

/**
 * Map a chart spec to an ECharts option.
 *
 * Returns the legacy `option` untouched when the archive carries one, and
 * `null` for a spec of unknown kind so the caller can render nothing rather
 * than an empty frame.
 */
export function renderReportChart(spec) {
  if (!spec || typeof spec !== 'object') return null
  if (spec.option && typeof spec.option === 'object') return spec.option
  const render = RENDERERS[spec.kind]
  return render ? render(spec) : null
}
