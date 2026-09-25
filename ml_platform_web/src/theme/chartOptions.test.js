import { describe, expect, it } from 'vitest'
import { darkChartOption } from './chartOptions'
import { colors } from './tokens'

describe('Dark chart presentation adapter', () => {
  it('preserves data, callbacks and semantic series colors without mutation', () => {
    const data = Object.freeze([1, 2, 3])
    const formatter = p => p.value
    const source = Object.freeze({ series: [{ type: 'line', data, itemStyle: { color: '#e34d59' } }], tooltip: { formatter } })
    const result = darkChartOption(source)
    expect(result.series[0].data).toBe(data)
    expect(result.series[0].itemStyle.color).toBe('#e34d59')
    expect(result.tooltip.formatter).toBe(formatter)
    expect(source.tooltip.backgroundColor).toBeUndefined()
  })
  it('makes legacy axes and tooltips readable on a dark surface', () => {
    const result = darkChartOption({ xAxis: { axisLabel: { color: '#334155' }, splitLine: { lineStyle: { color: '#e2e8f0' } } }, backgroundColor: '#ffffff' })
    expect(result.xAxis.axisLabel.color).toBe(colors.text)
    expect(result.xAxis.splitLine.lineStyle.color).toBe(colors.border)
    expect(result.tooltip.backgroundColor).toBe(colors.elevated)
    expect(result.backgroundColor).toBe(colors.surface)
  })
  it('resolves CSS tokens for canvas while preserving typed datasets', () => {
    const source = new Float32Array([1, 2])
    const result = darkChartOption({ series: [{ data: source }], title: { textStyle: { color: 'var(--text-primary)' } } }, () => '#eeeeee')
    expect(result.title.textStyle.color).toBe('#eeeeee')
    expect(result.series[0].data).toBe(source)
  })
})
