import { describe, expect, it } from 'vitest'

import {
  buildCurl,
  buildRequestExample,
  buildResponseExample,
  deploymentNotes,
  normaliseWeights,
  predictUrl,
  requestFeatureNames,
  suggestWeights,
} from './deploySchema'

describe('requestFeatureNames', () => {
  it('drops the target column — it is the thing being predicted', () => {
    expect(requestFeatureNames(['a', 'b', 'load'], 'load')).toEqual(['a', 'b'])
  })

  it('drops columns derived from the target', () => {
    // These are produced by the platform, not supplied by a caller; leaving
    // them in the example invites leakage-shaped requests.
    expect(requestFeatureNames(['a', 'predicted_load', 'prediction_x'], 'load')).toEqual(['a'])
  })
})

describe('buildRequestExample', () => {
  it('uses the real sample row so the example matches the caller data', () => {
    const out = buildRequestExample({
      sampleRow: { temp: 21.5, hour: 8, load: 9000 },
      columnNames: ['temp', 'hour', 'load'],
      targetColumn: 'load',
    })
    expect(out.rows).toEqual([{ temp: 21.5, hour: 8 }])
    expect(out.include_probabilities).toBe(true)
  })

  it('falls back per dtype for columns the sample lacks', () => {
    const out = buildRequestExample({
      sampleRow: {},
      columnsInfo: {
        temp: { dtype: 'float64', mean: 20.25 },
        hour: { dtype: 'int64', mean: 11.6 },
        flag: { dtype: 'bool' },
        city: { dtype: 'object' },
      },
      targetColumn: 'load',
    })
    expect(out.rows[0]).toEqual({ temp: 20.25, hour: 12, flag: false, city: '示例值' })
  })

  it('keeps a falsy sampled value instead of overwriting it', () => {
    // 0 and "" are legitimate readings; treating them as missing would show a
    // mean where the caller's data really has a zero.
    const out = buildRequestExample({
      sampleRow: { temp: 0 },
      columnsInfo: { temp: { dtype: 'float64', mean: 20 } },
      targetColumn: 'load',
    })
    expect(out.rows[0].temp).toBe(0)
  })
})

describe('buildResponseExample', () => {
  it('returns null probabilities for regression', () => {
    const out = buildResponseExample({ taskType: 'regression' })
    expect(out.probabilities).toBeNull()
    expect(typeof out.predictions[0]).toBe('number')
  })

  it('returns a per-class probability map for classification', () => {
    const out = buildResponseExample({ taskType: 'classification', classLabels: ['setosa', 'virginica'] })
    expect(out.predictions).toEqual(['setosa'])
    expect(Object.keys(out.probabilities[0])).toEqual(['setosa', 'virginica'])
  })
})

describe('predictUrl / buildCurl', () => {
  it('builds the endpoint from the given origin', () => {
    expect(predictUrl('dep-1', 'https://example.com'))
      .toBe('https://example.com/inference/dep-1/predict')
  })

  it('leaves a placeholder when there is no deployment yet', () => {
    expect(predictUrl(null, 'https://example.com'))
      .toBe('https://example.com/inference/{deployment_id}/predict')
  })

  it('emits a curl carrying the auth header and the body', () => {
    const curl = buildCurl({
      deploymentId: 'dep-1',
      origin: 'https://example.com',
      requestExample: { rows: [{ a: 1 }], include_probabilities: true },
    })
    expect(curl).toContain("https://example.com/inference/dep-1/predict")
    expect(curl).toContain('Authorization: Bearer')
    expect(curl).toContain('"rows":[{"a":1}]')
  })
})

describe('deploymentNotes', () => {
  it('warns about string encoding only when there are text features', () => {
    const withText = deploymentNotes({ hasTextFeatures: true }).join('\n')
    const withoutText = deploymentNotes({ hasTextFeatures: false }).join('\n')
    expect(withText).toContain('原始字符串')
    expect(withoutText).not.toContain('原始字符串')
  })

  it('states the configured batch limit', () => {
    expect(deploymentNotes({ maxBatchSize: 250 })[0]).toContain('250')
  })
})

describe('suggestWeights', () => {
  it('favours the higher score when higher is better', () => {
    const w = suggestWeights([
      { run_id: 'a', objective_value: 0.9 },
      { run_id: 'b', objective_value: 0.3 },
    ], 'max')
    expect(w.a).toBeGreaterThan(w.b)
    expect(w.a + w.b).toBeCloseTo(1)
  })

  it('favours the lower score when lower is better', () => {
    // rmse: 50 is twice as good as 100, so it should get twice the weight.
    const w = suggestWeights([
      { run_id: 'a', objective_value: 50 },
      { run_id: 'b', objective_value: 100 },
    ], 'min')
    expect(w.a).toBeCloseTo(2 / 3)
    expect(w.b).toBeCloseTo(1 / 3)
  })

  it('splits evenly when no member has a score', () => {
    const w = suggestWeights([{ run_id: 'a' }, { run_id: 'b' }], 'max')
    expect(w.a).toBeCloseTo(0.5)
    expect(w.b).toBeCloseTo(0.5)
  })

  it('always sums to 1', () => {
    const w = suggestWeights([
      { run_id: 'a', objective_value: 0.91 },
      { run_id: 'b', objective_value: 0.88 },
      { run_id: 'c', objective_value: 0.85 },
    ], 'max')
    expect(Object.values(w).reduce((x, y) => x + y, 0)).toBeCloseTo(1)
  })
})

describe('normaliseWeights', () => {
  it('rescales edited weights back to 1 while keeping their ratio', () => {
    const w = normaliseWeights({ a: 3, b: 1 })
    expect(w.a).toBeCloseTo(0.75)
    expect(w.b).toBeCloseTo(0.25)
  })

  it('drops zero and negative weights rather than propagating them', () => {
    const w = normaliseWeights({ a: 1, b: 0, c: -2 })
    expect(w).toEqual({ a: 1 })
  })
})
