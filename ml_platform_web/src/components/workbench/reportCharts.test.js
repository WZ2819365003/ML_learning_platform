import { describe, expect, it } from 'vitest'

import {
  REPORT_CHART_KINDS,
  REPORT_CHART_THEME,
  categoryAxisLeft,
  chartHeight,
  estimateTextWidth,
  formatTooltipRows,
  formatValue,
  renderReportChart,
} from './reportCharts'
import {
  allSpecs,
  fieldCompositionSpec,
  foldDotsSpec,
  leaderboardSpec,
  legacyOptionChart,
  lossHistorySpec,
  scatterPairSpec,
  shapBarsSpec,
  specsByKind,
  targetHistSpec,
} from './__fixtures__/reportCharts.fixtures'

const seriesTypes = (option) => (option.series || []).map((s) => s.type)

describe('renderReportChart · series per kind', () => {
  it('covers every kind in the contract with a fixture', () => {
    expect(Object.keys(specsByKind).sort()).toEqual([...REPORT_CHART_KINDS].sort())
    allSpecs.forEach((spec) => expect(renderReportChart(spec)).not.toBeNull())
  })

  it('hbar: horizontal bars per series, custom error bars, red dashed reference line', () => {
    const option = renderReportChart(leaderboardSpec)
    expect(seriesTypes(option)).toEqual(['bar', 'bar', 'custom'])
    expect(option.yAxis.type).toBe('category')
    expect(option.yAxis.inverse).toBe(true)
    expect(option.xAxis.type).toBe('value')
    // Only the cross-validation series carries fold spread; the hold-out one has error: null.
    expect(option.series[2].id).toBe('error-0')
    expect(option.series[2].encode).toEqual({ x: [0, 1], y: 2 })
    expect(option.series[2].data[0]).toEqual([72.4673 - 3.1, 72.4673 + 3.1, 0])
    expect(option.series[2].data[4]).toEqual([null, null, 4])
    const line = option.series[0].markLine
    expect(line.lineStyle).toMatchObject({ color: REPORT_CHART_THEME.colors.accent, type: 'dashed' })
    expect(line.data).toEqual([{ xAxis: 88.97, name: '平均负荷的 1%' }])
    expect(option.series[0].itemStyle.color).toBe(REPORT_CHART_THEME.colors.primary)
    expect(option.series[1].itemStyle.color).toBe(REPORT_CHART_THEME.colors.muted)
    expect(option.legend.show).toBe(true)
  })

  it('hbar without errors or references stays a single bar series', () => {
    const option = renderReportChart(shapBarsSpec)
    expect(seriesTypes(option)).toEqual(['bar'])
    expect(option.series[0].markLine).toBeUndefined()
    expect(option.legend.show).toBe(false)
  })

  it('dots: one scatter row per category plus a mean tick series', () => {
    const option = renderReportChart(foldDotsSpec)
    expect(seriesTypes(option)).toEqual(['scatter', 'scatter'])
    expect(option.series[0].data).toHaveLength(15)
    expect(option.series[0].data[0].value).toEqual([69.8, 0])
    expect(option.series[0].data[0].row).toEqual({ fold: 1, rmse: 69.8, mae: 52.1, r2: 0.983 })
    expect(option.series[1].id).toBe('mean')
    expect(option.series[1].symbol).toBe('rect')
    expect(option.series[1].data).toHaveLength(3)
    expect(option.series[1].data[0].value[0]).toBeCloseTo(72.46, 2)
  })

  it('hist: bars on a value axis spanning the bins, markers as dashed lines', () => {
    const option = renderReportChart(targetHistSpec)
    expect(seriesTypes(option)).toEqual(['bar'])
    expect(option.series[0].data).toHaveLength(20)
    expect(option.xAxis.min).toBe(5000)
    expect(option.xAxis.max).toBe(14274)
    expect(option.series[0].markLine.data.map((d) => d.name)).toEqual(['Q1', '均值', 'Q3'])
    expect(option.series[0].markLine.data[1].xAxis).toBe(8897)
  })

  it('stacked: one stacked bar segment per group on a single hidden category', () => {
    const option = renderReportChart(fieldCompositionSpec)
    expect(seriesTypes(option)).toEqual(['bar', 'bar', 'bar', 'bar', 'bar'])
    expect(option.series.every((s) => s.stack === 'total')).toBe(true)
    expect(option.yAxis.show).toBe(false)
    expect(option.xAxis.max).toBe(35)
    expect(option.series.map((s) => s.data[0].value)).toEqual([11, 8, 4, 3, 9])
  })

  it('lines: line per series, log y, best-epoch marker and early-stop shade', () => {
    const option = renderReportChart(lossHistorySpec)
    expect(seriesTypes(option)).toEqual(['line', 'line'])
    expect(option.yAxis.type).toBe('log')
    expect(option.series[0].data).toHaveLength(38)
    expect(option.series[0].data[30]).toEqual([31, lossHistorySpec.series[0].values[30]])
    expect(option.series[0].markLine.data).toEqual([{ xAxis: 31, name: '最优轮' }])
    expect(option.series[0].markArea.data[0][0]).toEqual({ xAxis: 31, name: '早停区' })
    expect(option.series[0].markArea.data[0][1]).toEqual({ xAxis: 38 })
    expect(option.tooltip.trigger).toBe('axis')
  })

  it('scatter_pair: two grids, actual/predicted lines on the left, residual bars on the right', () => {
    const option = renderReportChart(scatterPairSpec)
    expect(seriesTypes(option)).toEqual(['line', 'line', 'bar'])
    expect(option.grid).toHaveLength(2)
    expect(option.xAxis).toHaveLength(2)
    expect(option.yAxis).toHaveLength(2)
    expect(option.series[0].xAxisIndex).toBe(0)
    expect(option.series[2].xAxisIndex).toBe(1)
    expect(option.series[0].data).toHaveLength(120)
    expect(option.series[2].data).toHaveLength(12)
    expect(option.title[0].subtext).toBe('RMSE 72.5 · MAE 54.9 · RMSE/MAE 1.32')
  })
})

describe('renderReportChart · layout', () => {
  it('leaves room on the left for the longest category label', () => {
    const option = renderReportChart(leaderboardSpec)
    const longest = Math.max(...leaderboardSpec.categories.map((c) => estimateTextWidth(c)))
    expect(option.grid.left).toBeGreaterThanOrEqual(longest)
    expect(option.grid.containLabel).toBe(false)
  })

  it('estimates CJK glyphs as square and Latin as narrower', () => {
    expect(estimateTextWidth('交叉验证', 12)).toBe(48)
    expect(estimateTextWidth('xgboost', 12)).toBe(Math.ceil(7 * 12 * 0.58))
    expect(categoryAxisLeft(['a', '很长的中文标签'])).toBe(84 + 8 + 12)
  })

  it('wraps rather than clips a label longer than the cap', () => {
    const long = 'x'.repeat(60)
    const option = renderReportChart({ ...shapBarsSpec, categories: [long], rows: [], series: [{ name: 's', values: [1] }] })
    expect(option.grid.left).toBe(REPORT_CHART_THEME.maxLabelWidth + 20)
    expect(option.yAxis.axisLabel).toMatchObject({ width: 200, overflow: 'break' })
  })

  it('uses the theme height, with the documented stacked exception', () => {
    expect(chartHeight(leaderboardSpec)).toBe(300)
    expect(chartHeight(fieldCompositionSpec)).toBe(200)
    expect(chartHeight({ option: {}, height: 420 })).toBe(420)
  })

  it('reads all its colours from the theme roles', () => {
    const option = renderReportChart(leaderboardSpec)
    const colours = new Set([
      option.series[0].itemStyle.color, option.series[1].itemStyle.color, option.series[0].markLine.lineStyle.color,
    ])
    expect([...colours].sort()).toEqual([
      REPORT_CHART_THEME.colors.primary, REPORT_CHART_THEME.colors.muted, REPORT_CHART_THEME.colors.accent,
    ].sort())
  })
})

describe('renderReportChart · tooltips', () => {
  const labelsOf = (spec) => spec.tooltip_fields.map((f) => f.label)

  it('hbar tooltip lists every tooltip_field as label: value', () => {
    const option = renderReportChart(leaderboardSpec)
    const html = option.tooltip.formatter({ data: option.series[0].data[0] })
    labelsOf(leaderboardSpec).forEach((label) => expect(html).toContain(`${label}</span>: `))
    expect(html).toContain('RMSE</span>: 72.5')
    expect(html).toContain('占均值</span>: 0.8%')
    expect(html).toContain('R²</span>: 0.981')
    expect(html).toContain('验证方式</span>: 交叉验证')
  })

  it('hbar tooltip shows — for a missing value instead of NaN', () => {
    const option = renderReportChart(leaderboardSpec)
    const html = option.tooltip.formatter({ data: option.series[1].data[4] })
    expect(html).toContain('折间标准差</span>: —')
    expect(html).not.toContain('NaN')
  })

  it('dots tooltip shows the fold row of the hovered point', () => {
    const option = renderReportChart(foldDotsSpec)
    const html = option.tooltip.formatter({ seriesId: 'dots', data: option.series[0].data[7] })
    labelsOf(foldDotsSpec).forEach((label) => expect(html).toContain(label))
    expect(html).toContain('第几折</span>: 3')
    expect(html).toContain('RMSE</span>: 72.6')
    const mean = option.tooltip.formatter({ seriesId: 'mean', data: option.series[1].data[0] })
    expect(mean).toContain('均值')
    expect(mean).toContain('xgboost_regressor')
  })

  it('hist tooltip shows range, count and share', () => {
    const option = renderReportChart(targetHistSpec)
    const html = option.tooltip.formatter({ data: option.series[0].data[19] })
    labelsOf(targetHistSpec).forEach((label) => expect(html).toContain(label))
    expect(html).toContain('样本数</span>: 9')
    expect(html).toContain('区间</span>: 13810.3 – 14274')
    expect(html).toContain('占比</span>: 0.01%')
  })

  it('stacked tooltip lists the column names of the hovered segment', () => {
    const option = renderReportChart(fieldCompositionSpec)
    const html = option.tooltip.formatter({ data: option.series[2].data[0] })
    labelsOf(fieldCompositionSpec).forEach((label) => expect(html).toContain(label))
    expect(html).toContain('load_lag_1<br/>load_lag_2<br/>load_lag_24<br/>load_lag_168')
    expect(html).toContain('列数</span>: 4')
    expect(html).not.toContain('等 ')
  })

  it('stacked tooltip caps the list at 20 names and notes the total', () => {
    const items = Array.from({ length: 27 }, (_, i) => `feature_${i + 1}`)
    const option = renderReportChart({
      ...fieldCompositionSpec,
      segments: [{ name: '气象衍生', count: 27, items }],
    })
    const html = option.tooltip.formatter({ data: option.series[0].data[0] })
    expect(html).toContain('feature_20')
    expect(html).not.toContain('feature_21')
    expect(html).toContain('等 27 列')
  })

  it('lines tooltip shows the epoch row from rows', () => {
    const option = renderReportChart(lossHistorySpec)
    const html = option.tooltip.formatter([
      { seriesName: '训练损失', dataIndex: 30, value: [31, 0.005] },
      { seriesName: '验证损失', dataIndex: 30, value: [31, 0.006] },
    ])
    labelsOf(lossHistorySpec).forEach((label) => expect(html).toContain(label))
    expect(html).toContain('轮</span>: 31')
  })

  it('lines tooltip falls back to series values when the spec has no rows', () => {
    const option = renderReportChart({ ...lossHistorySpec, rows: [], tooltip_fields: [] })
    const html = option.tooltip.formatter([{ seriesName: '训练损失', dataIndex: 0, value: [1, 0.0412] }])
    expect(html).toContain('训练损失</span>: 0.0412')
    expect(html).toContain('epoch 1')
  })

  it('scatter_pair tooltip reads the pair row on the left and the bin on the right', () => {
    const option = renderReportChart(scatterPairSpec)
    const left = option.tooltip.formatter([{ seriesId: 'actual', dataIndex: 5 }, { seriesId: 'predicted', dataIndex: 5 }])
    labelsOf(scatterPairSpec).forEach((label) => expect(left).toContain(label))
    expect(left).toContain(`实际</span>: ${scatterPairSpec.pair.actual[5]}`)
    const right = option.tooltip.formatter([{ seriesId: 'residual', dataIndex: 0, data: option.series[2].data[0] }])
    expect(right).toContain('区间</span>: -120 – -100')
    expect(right).toContain('样本数')
  })

  it('escapes markup in labels and values', () => {
    const html = formatTooltipRows([{ key: 'a', label: '<b>' }], { a: '<img>' })
    expect(html).toContain('&lt;b&gt;')
    expect(html).toContain('&lt;img&gt;')
    expect(html).not.toContain('<img>')
  })
})

describe('formatValue', () => {
  it('defaults numbers to 4 decimals and keeps integers whole', () => {
    expect(formatValue(72.4673123)).toBe('72.4673')
    expect(formatValue(87312)).toBe('87312')
    expect(formatValue('xgboost')).toBe('xgboost')
    expect(formatValue(null)).toBe('—')
  })

  it('honours the field format', () => {
    expect(formatValue(72.4673, '.1f')).toBe('72.5')
    expect(formatValue(0.0081, 'pct')).toBe('0.8%')
    expect(formatValue(0.81, 'percent')).toBe('0.8%')
    expect(formatValue(100, 'percent')).toBe('100%')
    expect(formatValue(3.6, 'int')).toBe('4')
    expect(formatValue(1.23456, '2')).toBe('1.23')
    expect(formatValue(1.23456, '0.000')).toBe('1.235')
  })
})

describe('renderReportChart · legacy and unknown', () => {
  it('passes a legacy ECharts option through untouched', () => {
    expect(renderReportChart(legacyOptionChart)).toBe(legacyOptionChart.option)
  })

  it('returns null for an unknown kind or garbage so the caller renders nothing', () => {
    expect(renderReportChart({ id: 'x', kind: 'pie' })).toBeNull()
    expect(renderReportChart(null)).toBeNull()
    expect(renderReportChart('nope')).toBeNull()
  })

  it('renders an empty spec of a known kind without throwing', () => {
    REPORT_CHART_KINDS.forEach((kind) => {
      expect(() => renderReportChart({ id: kind, kind })).not.toThrow()
    })
  })
})
