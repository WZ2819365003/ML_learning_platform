import { describe, expect, it } from 'vitest'

import { crossValidationStability, earlyStoppingSummary } from './TrainingProcessPanel'

describe('crossValidationStability', () => {
  it('pairs each cv_avg_ metric with its std', () => {
    const rows = crossValidationStability({
      cv_avg_rmse: 100, cv_std_rmse: 5,
      cv_avg_r2: 0.9, cv_std_r2: 0.02,
    })
    const rmse = rows.find(r => r.metric === 'RMSE')
    expect(rmse.mean).toBe(100)
    expect(rmse.std).toBe(5)
    expect(rmse.cv).toBeCloseTo(5)   // 5/100
  })

  it('ignores non-cv metrics and non-numbers', () => {
    const rows = crossValidationStability({
      rmse: 72, cv_avg_rmse: 100, cv_std_rmse: 5, cv_avg_junk: 'nope',
    })
    expect(rows.map(r => r.metric)).toEqual(['RMSE'])
  })

  it('leaves cv null when the std is missing rather than guessing', () => {
    const rows = crossValidationStability({ cv_avg_mae: 10 })
    expect(rows[0].std).toBeNull()
    expect(rows[0].cv).toBeNull()
  })

  it('does not divide by a zero mean', () => {
    // An r2 that averaged to zero would otherwise produce Infinity.
    const rows = crossValidationStability({ cv_avg_r2: 0, cv_std_r2: 0.1 })
    expect(rows[0].cv).toBeNull()
  })

  it('returns nothing when there are no cv aggregates', () => {
    expect(crossValidationStability({ rmse: 72 })).toEqual([])
  })
})

describe('earlyStoppingSummary', () => {
  it('flags a run that stopped before its epoch budget', () => {
    const s = earlyStoppingSummary(
      { final_epoch: 18, best_val_loss: 0.42, history: new Array(18) },
      { epochs: 50 },
    )
    expect(s.ran).toBe(18)
    expect(s.budget).toBe(50)
    expect(s.stoppedEarly).toBe(true)
    expect(s.bestValLoss).toBe(0.42)
  })

  it('does not flag a run that used its whole budget', () => {
    const s = earlyStoppingSummary({ final_epoch: 50, history: new Array(50) }, { epochs: 50 })
    expect(s.stoppedEarly).toBe(false)
  })

  it('falls back to the history length when final_epoch is absent', () => {
    const s = earlyStoppingSummary({ history: new Array(7) }, { epochs: 20 })
    expect(s.ran).toBe(7)
  })

  it('never claims early stopping without a known budget', () => {
    // No configured epochs means there is nothing to have stopped short of.
    const s = earlyStoppingSummary({ final_epoch: 7, history: new Array(7) }, {})
    expect(s.stoppedEarly).toBe(false)
  })

  it('returns null when nothing ran', () => {
    expect(earlyStoppingSummary({}, {})).toBeNull()
  })
})
