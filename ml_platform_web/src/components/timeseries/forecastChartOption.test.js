import { describe, expect, it } from 'vitest'
import { buildOption, buildSeries, isUsableTimeline } from './forecastChartOption'

// 半小时粒度的 hour_of_day：0.0, 0.5, … 23.5 循环，正是线上那份电力负荷数据。
function cyclicAxis(length) {
  return Array.from({ length }, (_, i) => ((i % 48) / 2).toFixed(1))
}

function result({ history = 200, horizon = 24, axis = null } = {}) {
  return {
    historical: Array.from({ length: history }, (_, i) => 1000 + i),
    point_forecast: Array.from({ length: horizon }, (_, i) => 2000 + i),
    q10: Array.from({ length: horizon }, (_, i) => 1900 + i),
    q90: Array.from({ length: horizon }, (_, i) => 2100 + i),
    time_axis: { historical: axis, forecast: null },
  }
}

describe('isUsableTimeline', () => {
  it('接受基本唯一的时间戳列', () => {
    expect(isUsableTimeline(['2026-01-01', '2026-01-02', '2026-01-03'])).toBe(true)
  })

  it('拒绝周期列——这是 x 轴显示一整排 "0.5" 的根因', () => {
    expect(isUsableTimeline(cyclicAxis(480))).toBe(false)
  })

  it('空列或缺失列不当时间轴', () => {
    expect(isUsableTimeline([])).toBe(false)
    expect(isUsableTimeline(null)).toBe(false)
  })
})

describe('buildSeries', () => {
  it('周期时间列改用相对步标签，原值仍保留给悬停窗', () => {
    const series = buildSeries(result({ history: 480, axis: cyclicAxis(480) }), 96)
    expect(series.timeline).toBe(false)
    expect(series.labels[0]).toBe('t-96')
    expect(series.labels[95]).toBe('t-1')
    expect(series.labels[96]).toBe('t+1')
    // 轴标签互不重复，抽稀之后不会变成一排相同的数字
    expect(new Set(series.labels).size).toBe(series.labels.length)
    expect(series.historySource).toHaveLength(96)
  })

  it('真实时间戳列直接当标签', () => {
    const axis = Array.from({ length: 50 }, (_, i) => `2026-01-${String(i + 1).padStart(2, '0')}`)
    const series = buildSeries(result({ history: 50, axis }), 10)
    expect(series.timeline).toBe(true)
    expect(series.labels[0]).toBe('2026-01-41')
  })

  it('窗口只截历史尾部，预测段一个不少', () => {
    const series = buildSeries(result({ history: 87312, horizon: 24 }), 144)
    expect(series.history).toHaveLength(144)
    expect(series.forecast).toHaveLength(24)
    expect(series.splitIndex).toBe(144)
    expect(series.totalHistory).toBe(87312)
    // 预测段占了轴的 1/7，而不是 87336 分之 24
    expect(series.forecast.length / series.labels.length).toBeGreaterThan(0.1)
  })

  it('窗口为 Infinity 时保留全部历史', () => {
    const series = buildSeries(result({ history: 500 }), Infinity)
    expect(series.history).toHaveLength(500)
  })

  it('窗口大于历史长度时不越界', () => {
    const series = buildSeries(result({ history: 30 }), 5000)
    expect(series.history).toHaveLength(30)
    expect(series.splitIndex).toBe(30)
  })
})

describe('buildOption', () => {
  const option = buildOption(buildSeries(result({ history: 200, horizon: 24 }), 144))
  const byName = name => option.series.filter(s => s.name === name)

  it('置信带上沿画在 q90，而不是 q90 + q10', () => {
    const [lower, upper] = byName('Q10 ~ Q90')
    // 两条堆叠：下界打底 1900，上界只堆差值 200 → 叠加后正好 2100 = q90
    expect(lower.data[144]).toBe(1900)
    expect(upper.data[144]).toBe(200)
    expect(lower.data[144] + upper.data[144]).toBe(2100)
  })

  it('预测线从历史最后一点接上，不悬空', () => {
    const [forecastLine] = byName('预测值')
    expect(forecastLine.data[143]).toBe(1199) // 历史最后一点
    expect(forecastLine.data[144]).toBe(2000) // 预测第一点
    expect(forecastLine.data[142]).toBeNull()
  })

  it('预测段的标注挂在预测线上——历史线在这一段全是 null，挂上去不会渲染', () => {
    const [history] = byName('历史值')
    const [forecastLine] = byName('预测值')
    expect(history.markArea).toBeUndefined()
    expect(forecastLine.markLine.data[0].xAxis).toBe('t+1')
    expect(forecastLine.markArea.data[0][0].xAxis).toBe('t+1')
    expect(forecastLine.markArea.data[0][1].xAxis).toBe('t+24')
  })

  it('点数超阈值时降级为 LTTB 采样', () => {
    const dense = buildOption(buildSeries(result({ history: 87312 }), Infinity))
    expect(dense.series[0].sampling).toBe('lttb')
    expect(dense.series[0].smooth).toBe(false)
    expect(dense.animation).toBe(false)
  })

  it('滚轮默认留给页面，Shift 才缩放', () => {
    expect(option.dataZoom[0].zoomOnMouseWheel).toBe('shift')
    expect(option.dataZoom[0].moveOnMouseWheel).toBe(false)
  })

  it('缺少分位数时不画置信带', () => {
    const noBand = buildOption(buildSeries({
      historical: [1, 2, 3],
      point_forecast: [4, 5],
      q10: [],
      q90: [],
    }, 3))
    expect(noBand.series.filter(s => s.name === 'Q10 ~ Q90')).toHaveLength(0)
    expect(noBand.legend.data).toEqual(['历史值', '预测值'])
  })
})
