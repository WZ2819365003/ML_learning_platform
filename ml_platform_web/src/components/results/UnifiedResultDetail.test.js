import { describe, expect, it } from 'vitest'

import { normalizeTrainConfig } from './UnifiedResultDetail'

describe('normalizeTrainConfig', () => {
  it('lifts the ML hold-out ratio into the shared config shape', () => {
    expect(normalizeTrainConfig({ test_size: 0.2 })).toEqual({ test_size: 0.2 })
  })

  it('preserves DL config and fills the epoch budget from task status', () => {
    expect(normalizeTrainConfig({
      train_config: { batch_size: 64, test_size: 0.25 },
      total_epochs: 50,
    })).toEqual({ batch_size: 64, test_size: 0.25, epochs: 50 })
  })

  it('does not overwrite explicit nested values', () => {
    expect(normalizeTrainConfig({
      train_config: { test_size: 0.3, epochs: 20 },
      test_size: 0.2,
      total_epochs: 50,
    })).toEqual({ test_size: 0.3, epochs: 20 })
  })
})
