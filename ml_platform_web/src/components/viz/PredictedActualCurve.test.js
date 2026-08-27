import { describe, expect, it } from 'vitest'

import { buildPredictedActualOption } from './PredictedActualCurve'

describe('buildPredictedActualOption', () => {
  it('plots actual and predicted as two aligned series', () => {
    const o = buildPredictedActualOption([1, 2, 3], [1.1, 2.1, 2.9], 200)
    expect(o.series.map(s => s.name)).toEqual(['实际值', '预测值'])
    expect(o.series[0].data).toEqual([1, 2, 3])
    expect(o.series[1].data).toEqual([1.1, 2.1, 2.9])
  })

  it('shows the most recent window, not the first N', () => {
    // A forecast is judged on where it ended up, so the tail is the useful end.
    const actual = Array.from({ length: 10 }, (_, i) => i)
    const o = buildPredictedActualOption(actual, actual, 3)
    expect(o.series[0].data).toEqual([7, 8, 9])
    expect(o.xAxis.data).toEqual([8, 9, 10])
  })

  it('never plots past the shorter of the two arrays', () => {
    // Mismatched lengths would otherwise pad one line with undefined and draw
    // a break that looks like missing data rather than a payload problem.
    const o = buildPredictedActualOption([1, 2, 3, 4], [1, 2], 200)
    expect(o.series[0].data).toHaveLength(2)
    expect(o.series[1].data).toHaveLength(2)
  })

  it('caps the window at the data length', () => {
    const o = buildPredictedActualOption([1, 2], [1, 2], 1000)
    expect(o.series[0].data).toEqual([1, 2])
    expect(o.xAxis.data).toEqual([1, 2])
  })
})
