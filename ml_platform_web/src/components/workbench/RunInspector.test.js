import { describe, expect, it } from 'vitest'

import { DASH, deriveRunFields, formatProgress } from './RunInspector'

// A payload shaped like GET /api/platform/runs/{id}/inspector for a DL run:
// training_task is the DL serializer's output, which carries no target_column.
const dlPayload = {
  run: { id: 'r1', trial_no: 2, rank: 1, status: 'SUCCESS', metrics: { rmse: 4.2 } },
  experiment: { id: 'e1', name: 'grid', objective_metric: 'rmse' },
  training_task: {
    id: 't1', family: 'dl', model_type: 'mlp_dl', task_type: 'regression',
    status: 'SUCCESS', progress: 100, model_path: '/tmp/m.pt',
    dataset: { id: 'd1', name: 'pv_generation.csv', row_count: 8760, column_count: 12 },
  },
  modeling_task: {
    id: 'm1', name: '光伏出力预测', target_column: 'ac_power',
    task_type: 'regression', objective_metric: 'rmse',
    objective_direction: 'min', dataset_name: 'pv_generation.csv', rank: 3,
  },
}

describe('formatProgress', () => {
  it('treats progress as an already-scaled percentage', () => {
    // The regression this guards: the drawer used to render `progress * 100`,
    // turning a finished run's stored 100 into "10000%".
    expect(formatProgress(100)).toBe('100%')
    expect(formatProgress(0)).toBe('0%')
    expect(formatProgress(42.4)).toBe('42%')
    expect(formatProgress(66.6)).toBe('67%')
  })

  it('falls back to a dash for missing or non-numeric progress', () => {
    expect(formatProgress(null)).toBe(DASH)
    expect(formatProgress(undefined)).toBe(DASH)
    expect(formatProgress('not-a-number')).toBe(DASH)
    expect(formatProgress(NaN)).toBe(DASH)
  })

  it('keeps 0 distinguishable from missing', () => {
    expect(formatProgress(0)).not.toBe(DASH)
  })
})

describe('deriveRunFields', () => {
  it('reads the modeling contract off modeling_task', () => {
    const f = deriveRunFields(dlPayload)
    expect(f.modelingTaskName).toBe('光伏出力预测')
    expect(f.targetColumn).toBe('ac_power')
    expect(f.datasetFile).toBe('pv_generation.csv')
    expect(f.progressLabel).toBe('100%')
  })

  it('keeps the task rank and the experiment rank apart', () => {
    // run.rank is 1 *within its experiment*; the task leaderboard puts it 3rd.
    // Showing the former under a bare 「排名」 label was the original bug.
    const f = deriveRunFields(dlPayload)
    expect(f.taskRank).toBe(3)
    expect(f.experimentRank).toBe(1)
    expect(f.isTopOne).toBe(false)
  })

  it('marks Top-1 from the task leaderboard, not the experiment rank', () => {
    const f = deriveRunFields({
      ...dlPayload,
      run: { ...dlPayload.run, rank: 4 },
      modeling_task: { ...dlPayload.modeling_task, rank: 1 },
    })
    expect(f.isTopOne).toBe(true)
  })

  it('degrades to a dash when modeling_task is missing, without throwing', () => {
    const f = deriveRunFields({ ...dlPayload, modeling_task: null })
    expect(f.modelingTaskName).toBe(DASH)
    expect(f.datasetFile).toBe(DASH)
    expect(f.taskRank).toBe(DASH)
    // DL training_task carries no target_column, so there is nothing to fall
    // back to — a dash beats inventing one.
    expect(f.targetColumn).toBe(DASH)
    // ...but everything training_task *does* own still resolves.
    expect(f.experimentRank).toBe(1)
    expect(f.progressLabel).toBe('100%')
  })

  it('shows a null leaderboard rank as a dash rather than a made-up position', () => {
    // A RUNNING/FAILED run has no objective value, so the backend sends rank:null.
    const f = deriveRunFields({
      ...dlPayload,
      run: { ...dlPayload.run, rank: null, status: 'RUNNING' },
      modeling_task: { ...dlPayload.modeling_task, rank: null },
    })
    expect(f.taskRank).toBe(DASH)
    expect(f.experimentRank).toBe(DASH)
    expect(f.isTopOne).toBe(false)
    expect(f.targetColumn).toBe('ac_power')  // contract still known
  })

  it('falls back to training_task.target_column for legacy ML runs', () => {
    // An ML run whose experiment predates the ModelingTask link: the legacy row
    // holds a correct target column, so dropping it would lose information.
    const f = deriveRunFields({
      run: { id: 'r2', rank: 2 },
      training_task: { family: 'ml', target_column: 'y', progress: 100 },
      modeling_task: null,
    })
    expect(f.targetColumn).toBe('y')
  })

  it('prefers modeling_task.target_column when both are present', () => {
    const f = deriveRunFields({
      ...dlPayload,
      training_task: { ...dlPayload.training_task, target_column: 'stale_col' },
    })
    expect(f.targetColumn).toBe('ac_power')
  })

  it('survives an empty, partial, or absent payload', () => {
    for (const payload of [undefined, null, {}, { run: null }, { training_task: {} }]) {
      const f = deriveRunFields(payload)
      expect(f.targetColumn).toBe(DASH)
      expect(f.taskRank).toBe(DASH)
      expect(f.experimentRank).toBe(DASH)
      expect(f.datasetFile).toBe(DASH)
      expect(f.modelingTaskName).toBe(DASH)
      expect(f.progressLabel).toBe(DASH)
      expect(f.isTopOne).toBe(false)
    }
  })
})
