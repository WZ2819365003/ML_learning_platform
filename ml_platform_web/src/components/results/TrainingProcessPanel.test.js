import { describe, expect, it } from 'vitest'

import {
  buildFoldScoresOption,
  buildLearningRateOption,
  buildLossOption,
  buildMetricHistoryOption,
  buildOverfitGapOption,
  crossValidationStability,
  earlyStoppingSummary,
} from './TrainingProcessPanel'

const HISTORY = [
  { epoch: 1, total: 50, train_loss: 1.0, val_loss: 1.2, lr: 0.001, val_rmse: 120 },
  { epoch: 2, total: 50, train_loss: 0.6, val_loss: 0.8, lr: 0.001, val_rmse: 90 },
  { epoch: 3, total: 50, train_loss: 0.3, val_loss: 0.9, lr: 0.0005, val_rmse: 95 },
]

describe('crossValidationStability', () => {
  it('pairs each cv_avg_ metric with its std', () => {
    const rows = crossValidationStability({ cv_avg_rmse: 100, cv_std_rmse: 5 })
    expect(rows[0]).toMatchObject({ metric: 'RMSE', mean: 100, std: 5 })
    expect(rows[0].cv).toBeCloseTo(5)
  })

  it('leaves cv null rather than dividing by a zero mean', () => {
    expect(crossValidationStability({ cv_avg_r2: 0, cv_std_r2: 0.1 })[0].cv).toBeNull()
  })

  it('ignores non-cv keys and non-numbers', () => {
    const rows = crossValidationStability({ rmse: 72, cv_avg_junk: 'nope', cv_avg_mae: 1 })
    expect(rows.map(r => r.metric)).toEqual(['MAE'])
  })
})

describe('earlyStoppingSummary', () => {
  it('flags a run that stopped short of its budget', () => {
    const s = earlyStoppingSummary({ final_epoch: 18, history: new Array(18) }, { epochs: 50 })
    expect(s).toMatchObject({ ran: 18, budget: 50, stoppedEarly: true })
  })

  it('never claims early stopping without a known budget', () => {
    expect(earlyStoppingSummary({ final_epoch: 7, history: new Array(7) }, {}).stoppedEarly).toBe(false)
  })

  it('returns null when nothing ran', () => {
    expect(earlyStoppingSummary({}, {})).toBeNull()
  })
})

describe('buildLossOption', () => {
  it('plots train and validation as two series', () => {
    const o = buildLossOption(HISTORY)
    expect(o.series.map(s => s.name)).toEqual(['训练损失', '验证损失'])
    expect(o.series[0].data).toEqual([1.0, 0.6, 0.3])
  })

  it('returns null when no row carries a loss', () => {
    // Better an explicit placeholder than an axis with nothing on it.
    expect(buildLossOption([{ epoch: 1, val_rmse: 9 }])).toBeNull()
    expect(buildLossOption([])).toBeNull()
  })
})

describe('buildOverfitGapOption', () => {
  it('plots validation minus training and marks the widest point', () => {
    const o = buildOverfitGapOption(HISTORY)
    // Element-wise approximate: these are float subtractions, so 1.2 - 1.0 is
    // 0.19999999999999996 and an exact match would be asserting on IEEE noise.
    expect(o.series[0].data[0]).toBeCloseTo(0.2)
    expect(o.series[0].data[1]).toBeCloseTo(0.2)
    expect(o.series[0].data[2]).toBeCloseTo(0.6)
    // Epoch 3 is where the two curves are furthest apart.
    expect(o.series[0].markPoint.data[0].coord[0]).toBe(2)
  })

  it('skips rows missing either loss rather than treating them as zero', () => {
    const o = buildOverfitGapOption([
      { epoch: 1, train_loss: 1, val_loss: 2 },
      { epoch: 2, train_loss: 1 },
    ])
    expect(o.series[0].data).toHaveLength(1)
  })

  it('returns null when no row has both', () => {
    expect(buildOverfitGapOption([{ epoch: 1, train_loss: 1 }])).toBeNull()
  })
})

describe('buildLearningRateOption', () => {
  it('uses a log axis so successive halvings stay visible', () => {
    // On a linear axis a scheduler's later steps collapse onto zero.
    expect(buildLearningRateOption(HISTORY).yAxis.type).toBe('log')
  })

  it('returns null when the trainer recorded no lr', () => {
    expect(buildLearningRateOption([{ epoch: 1, train_loss: 1 }])).toBeNull()
  })
})

describe('buildMetricHistoryOption', () => {
  it('plots validation metrics and leaves loss and lr to their own charts', () => {
    const names = buildMetricHistoryOption(HISTORY).series.map(s => s.name)
    expect(names).toEqual(['val_rmse'])
  })

  it('returns null when there is nothing but loss', () => {
    expect(buildMetricHistoryOption([{ epoch: 1, train_loss: 1, val_loss: 1, lr: 0.1 }])).toBeNull()
  })
})

describe('buildFoldScoresOption', () => {
  const FOLDS = [
    { fold: 1, rmse: 70 }, { fold: 2, rmse: 72 }, { fold: 3, rmse: 95 },
  ]

  it('draws one bar per fold with a mean line', () => {
    const o = buildFoldScoresOption(FOLDS)
    expect(o.series[0].data).toEqual([70, 72, 95])
    expect(o.series[0].markLine.data[0].yAxis).toBeCloseTo(79)
  })

  it('returns null when the trainer did not persist folds', () => {
    // Runs from before cv_folds was kept fall here, and get a placeholder.
    expect(buildFoldScoresOption([])).toBeNull()
    expect(buildFoldScoresOption(undefined)).toBeNull()
  })

  it('honours an explicitly chosen metric', () => {
    const o = buildFoldScoresOption([{ fold: 1, rmse: 70, r2: 0.9 }], 'r2')
    expect(o.series[0].data).toEqual([0.9])
  })
})
