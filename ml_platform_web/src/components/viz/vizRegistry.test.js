import { describe, expect, it } from 'vitest'

import { deriveRegressionViz, getVizEntries } from './vizRegistry'

describe('visualization registry', () => {
  it('loads only the shared regression prediction payload on the comparison tab', () => {
    const entries = getVizEntries({
      taskType: 'regression', family: 'ml', surface: 'results', tab: 'comparison',
    })
    expect(entries.map(entry => entry.key)).toEqual(['predictedVsActual'])
  })

  it('reuses the same registry in the ML workbench without duplicate residual prediction', () => {
    const entries = getVizEntries({
      taskType: 'regression', family: 'ml', surface: 'workbench',
    })
    expect(entries.map(entry => entry.key).sort()).toEqual([
      'featureImportance', 'learningCurve', 'predictedVsActual',
    ])
    expect(entries.some(entry => entry.key === 'shap')).toBe(false)
  })

  it('keeps classification-only charts out of regression routes', () => {
    const entries = getVizEntries({
      taskType: 'classification', family: 'ml', surface: 'results', tab: 'performance',
    })
    expect(entries.map(entry => entry.key)).toEqual([
      'confusionMatrix', 'rocCurve', 'perClass', 'prCurve',
    ])
    expect(entries.some(entry => entry.key === 'predictedVsActual')).toBe(false)
  })
})

describe('deriveRegressionViz', () => {
  it('derives residual scatter and a compatible histogram from one prediction payload', () => {
    const derived = deriveRegressionViz({
      actual: [10, 20, 30],
      predicted: [8, 21, 27],
      sample_count: 3,
      total_count: 100,
      sample_offset: 97,
      truncated: true,
    })

    expect(derived.residualPlot.residuals).toEqual([2, -1, 3])
    expect(derived.distribution.kind).toBe('regression_residuals')
    expect(derived.distribution.bin_edges).toHaveLength(31)
    expect(derived.distribution.counts.reduce((sum, count) => sum + count, 0)).toBe(3)
    expect(derived.distribution.total_count).toBe(100)
    expect(derived.distribution.sample_offset).toBe(97)
  })
})
