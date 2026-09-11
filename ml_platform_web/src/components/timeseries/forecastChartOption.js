// 预测图的纯数据逻辑：从组件里拆出来，方便单测，也避免 fast-refresh 抱怨
// 一个文件里同时导出组件和工具函数。
// 超过这个点数就不再逐点描线：LTTB 采样 + 关闭平滑，否则 8 万点会卡住主线程。
const SAMPLING_THRESHOLD = 3000

function toNumber(value) {
  const n = Number(value)
  return Number.isFinite(n) ? n : null
}

function formatNumber(value) {
  if (value == null || !Number.isFinite(Number(value))) return '—'
  return Number(value).toLocaleString('zh-CN', { maximumFractionDigits: 2 })
}

/**
 * 时间列能不能直接当坐标轴标签用。
 *
 * 像 hour_of_day 这种周期列，8 万行里只有 48 个不同取值，ECharts 按固定步长抽稀
 * 之后每个刻度都落在同一个相位上，整条轴会显示成一排一模一样的数字（例如全是
 * "0.5"）。这种列只能进悬停窗，不能当轴标签。
 */
export function isUsableTimeline(axis) {
  if (!axis?.length) return false
  const distinct = new Set(axis)
  return distinct.size >= axis.length * 0.9
}

/**
 * 按窗口裁出要画的那一段：历史尾部 + 全部预测。
 * 默认只留预测长度的几倍历史，否则 24 个预测点混在 8 万个历史点里根本看不见。
 */
export function buildSeries(result, windowSize) {
  const historical = (result?.historical ?? []).map(toNumber)
  const forecast = (result?.point_forecast ?? []).map(toNumber)
  const q10 = (result?.q10 ?? []).map(toNumber)
  const q90 = (result?.q90 ?? []).map(toNumber)
  const sourceAxis = result?.time_axis?.historical ?? null

  const keep = Number.isFinite(windowSize)
    ? Math.min(Math.max(windowSize, 1), historical.length)
    : historical.length
  const offset = historical.length - keep
  const history = historical.slice(offset)
  const historySource = sourceAxis?.length
    ? sourceAxis.slice(-historical.length).slice(offset)
    : null

  const timeline = isUsableTimeline(historySource)
  const historyLabels = timeline
    ? historySource
    : history.map((_, index) => `t-${keep - index}`)
  const forecastLabels = forecast.map((_, index) => `t+${index + 1}`)

  return {
    history,
    forecast,
    q10,
    q90,
    historySource,
    timeline,
    labels: [...historyLabels, ...forecastLabels],
    // 预测段第一个点在整条轴上的下标。
    splitIndex: keep,
    totalHistory: historical.length,
  }
}

export function buildOption(series) {
  const { history, forecast, q10, q90, labels, splitIndex } = series
  const gap = history.map(() => null)
  const dense = labels.length > SAMPLING_THRESHOLD
  const hasBand = q10.length === forecast.length && q90.length === forecast.length

  // 预测线从历史最后一点接上，避免虚线悬空一格。
  const bridged = [...gap.slice(0, -1), history[history.length - 1] ?? null, ...forecast]

  const tooltipRow = (marker, label, value) =>
    `<div style="display:flex;justify-content:space-between;gap:18px;line-height:1.8">`
    + `<span>${marker}${label}</span>`
    + `<span style="font-variant-numeric:tabular-nums;font-weight:500">${value}</span></div>`

  return {
    color: ['#1a8dff', '#18c3e3', '#ed7b2f'],
    animation: !dense,
    tooltip: {
      trigger: 'axis',
      confine: true,
      axisPointer: { type: 'line', lineStyle: { color: 'rgba(148,163,184,0.5)' } },
      formatter: (params) => {
        if (!params?.length) return ''
        const index = params[0].dataIndex
        const isForecast = index >= splitIndex
        const step = isForecast ? index - splitIndex : index
        const head = isForecast ? `预测第 ${step + 1} 步` : labels[index]
        // 时间列是周期列时不当轴标签，但仍然有参考价值，放进悬停窗。
        const source = !series.timeline && !isForecast ? series.historySource?.[index] : null

        let html = `<div style="font-weight:600;margin-bottom:6px;padding-bottom:5px;`
          + `border-bottom:1px solid rgba(148,163,184,0.25)">${head}</div>`
        if (source != null) html += tooltipRow('', '时间列', source)
        if (isForecast) {
          html += tooltipRow('', '预测值', formatNumber(forecast[step]))
          if (hasBand) {
            html += tooltipRow('', '置信区间', `${formatNumber(q10[step])} ~ ${formatNumber(q90[step])}`)
          }
        } else {
          html += tooltipRow('', '历史值', formatNumber(history[index]))
        }
        return html
      },
    },
    legend: {
      bottom: 0,
      itemGap: 18,
      data: hasBand ? ['历史值', '预测值', 'Q10 ~ Q90'] : ['历史值', '预测值'],
    },
    grid: { left: 64, right: 24, top: 28, bottom: 64 },
    // 滚轮默认留给页面滚动，按住 Shift 才缩放横轴。
    dataZoom: [{ type: 'inside', xAxisIndex: 0, zoomOnMouseWheel: 'shift', moveOnMouseWheel: false }],
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: labels,
      axisLabel: { hideOverlap: true },
    },
    yAxis: {
      type: 'value',
      scale: true,
      splitLine: { lineStyle: { color: 'rgba(148, 163, 184, 0.18)' } },
    },
    series: [
      {
        name: '历史值',
        type: 'line',
        smooth: !dense,
        symbol: 'none',
        sampling: dense ? 'lttb' : undefined,
        large: dense,
        lineStyle: { width: dense ? 1 : 2 },
        data: [...history, ...forecast.map(() => null)],
        markArea: {
          silent: true,
          itemStyle: { color: 'rgba(24, 195, 227, 0.07)' },
          label: { show: true, position: 'insideTop', color: '#18c3e3', fontSize: 12, formatter: '预测段' },
          data: [[{ xAxis: labels[splitIndex] }, { xAxis: labels[labels.length - 1] }]],
        },
        markLine: {
          silent: true,
          symbol: 'none',
          lineStyle: { color: '#18c3e3', type: 'dashed', width: 1 },
          label: { show: false },
          data: [{ xAxis: labels[splitIndex] }],
        },
      },
      {
        name: '预测值',
        type: 'line',
        smooth: true,
        symbol: 'none',
        lineStyle: { width: 2.5, type: 'dashed' },
        data: bridged,
        z: 4,
      },
      // 区间用两条堆叠线画：下界透明打底，上界只堆叠差值，填充出来的才是 Q10~Q90
      // 这一条带。原来两条都堆叠原值，上沿画到了 q90+q10。
      ...(hasBand
        ? [
          {
            name: 'Q10 ~ Q90',
            type: 'line',
            stack: 'band',
            symbol: 'none',
            silent: true,
            lineStyle: { opacity: 0 },
            areaStyle: { opacity: 0 },
            data: [...gap, ...q10],
            legendHoverLink: false,
            tooltip: { show: false },
          },
          {
            name: 'Q10 ~ Q90',
            type: 'line',
            stack: 'band',
            symbol: 'none',
            silent: true,
            lineStyle: { opacity: 0 },
            areaStyle: { color: 'rgba(237, 123, 47, 0.22)' },
            data: [...gap, ...q90.map((upper, index) => {
              const lower = q10[index]
              return upper == null || lower == null ? null : upper - lower
            })],
            legendHoverLink: false,
            tooltip: { show: false },
          },
        ]
        : []),
    ],
  }
}
