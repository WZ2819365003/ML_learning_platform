import { describe, expect, it } from 'vitest'

import { withReportChartDefaults } from './AiReportModal'

describe('withReportChartDefaults', () => {
  it('zooms the training metric y-axis around observed values', () => {
    const option = withReportChartDefaults({
      id: 'training_curves',
      option: {
        yAxis: { type: 'value', name: 'metric value' },
        series: [
          {
            type: 'line',
            data: [
              ['baseline#2', 0.9731],
              ['grid_search#3', 0.9764],
              ['bayesian_search#5', 0.979],
            ],
          },
        ],
      },
    })

    expect(option.yAxis.scale).toBe(true)
    expect(option.yAxis.min).toBeGreaterThan(0.9)
    expect(option.yAxis.min).toBeLessThan(0.9731)
    expect(option.yAxis.max).toBeGreaterThan(0.979)
    expect(option.yAxis.max).toBeLessThanOrEqual(1)
  })
})
