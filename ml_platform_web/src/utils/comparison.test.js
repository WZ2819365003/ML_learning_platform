import { describe, it, expect } from 'vitest'
import { buildComparisonVM } from './comparison'

const task = { objective_metric: 'accuracy', objective_direction: 'max' }
const rows = [
  { run_id: 'a', params: { model_type: 'random_forest' }, strategy_type: 'baseline', status: 'SUCCESS', metrics: { accuracy: 0.9, f1: 0.88 }, objective_value: 0.9, domain_task_id: 'd1', family: 'ml' },
  { run_id: 'b', params: { model_type: 'xgboost' }, strategy_type: 'baseline', status: 'SUCCESS', metrics: { accuracy: 0.96, roc_auc: 1.0 }, objective_value: 0.96, domain_task_id: 'd2', family: 'ml' },
]

describe('buildComparisonVM', () => {
  it('metricKeys = union of run metrics, objective first', () => {
    const vm = buildComparisonVM(rows, task)
    expect(vm.metricKeys[0]).toBe('accuracy')
    expect(new Set(vm.metricKeys)).toEqual(new Set(['accuracy', 'f1', 'roc_auc']))
  })
  it('is_best by direction=max', () => {
    const vm = buildComparisonVM(rows, task)
    expect(vm.rows.find(r => r.is_best).run_id).toBe('b')
  })
  it('is_best by direction=min', () => {
    const vm = buildComparisonVM(rows, { objective_metric: 'rmse', objective_direction: 'min' })
    expect(vm.rows.find(r => r.is_best).run_id).toBe('a')
  })
  it('model_type from params, carries domain_task_id/family', () => {
    const vm = buildComparisonVM(rows, task)
    expect(vm.rows[0].model_type).toBe('random_forest')
    expect(vm.rows[0].domain_task_id).toBe('d1')
  })
  it('empty rows → empty vm', () => {
    const vm = buildComparisonVM([], task)
    expect(vm.rows).toEqual([])
    expect(vm.metricKeys).toEqual(['accuracy'])
  })
})
