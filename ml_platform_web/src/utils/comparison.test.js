import { describe, it, expect } from 'vitest'
import {
  buildComparisonVM,
  buildFinalizationVM,
  buildStrategyCardVM,
  finalizeTaskAndRefresh,
} from './comparison'

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
    const row = vm.rows.find(r => r.run_id === 'a')
    expect(row.model_type).toBe('random_forest')
    expect(row.domain_task_id).toBe('d1')
  })
  it('empty rows → empty vm', () => {
    const vm = buildComparisonVM([], task)
    expect(vm.rows).toEqual([])
    expect(vm.metricKeys).toEqual(['accuracy'])
  })
  it('uses selection value for objective display and hides semantic aliases', () => {
    const vm = buildComparisonVM([{
      run_id: 'semantic',
      metrics: {
        accuracy: 0.99,
        cv_avg_accuracy: 0.81,
        selection_cv_mean_accuracy: 0.82,
        selection_cv_std_accuracy: 0.03,
        final_test_accuracy: 0.99,
        cv_folds: [{ fold: 1 }],
      },
      objective_value: 0.82,
      selection_metric_key: 'selection_cv_mean_accuracy',
      selection_value: 0.82,
      final_test_metric_key: 'final_test_accuracy',
      final_test_value: 0.99,
    }], task)

    expect(vm.rows[0].metrics.accuracy).toBe(0.82)
    expect(vm.rows[0].selection_metric_key).toBe('selection_cv_mean_accuracy')
    expect(vm.rows[0].final_test_value).toBe(0.99)
    expect(vm.metricKeys).toEqual(['accuracy', 'cv_avg_accuracy'])
  })

  it('resolves DL selection_val_* scores and hides them from metric columns (B1)', () => {
    const vm = buildComparisonVM([{
      run_id: 'dl-selection',
      status: 'SUCCESS',
      params: { model_type: 'mlp_dl', eval_metrics: ['accuracy', 'f1'] },
      metrics: {
        selection_val_accuracy: 0.87,
        selection_val_f1: 0.85,
        val_loss: 0.4,
      },
      selection_metric_key: 'selection_val_accuracy',
      selection_value: 0.87,
      domain_task_id: 'dl1',
      family: 'dl',
    }], task)

    expect(vm.rows[0].metrics.accuracy).toBe(0.87)
    expect(vm.rows[0].metrics.f1).toBe(0.85)
    expect(vm.metricKeys).not.toContain('selection_val_accuracy')
  })

  it('uses requested eval_metrics and maps CV aliases to display metrics', () => {
    const vm = buildComparisonVM([{
      run_id: 'requested-metrics',
      status: 'SUCCESS',
      params: {
        model_type: 'logistic_regression',
        eval_metrics: ['accuracy', 'f1', 'roc_auc'],
      },
      metrics: {
        cv_avg_accuracy: 0.81,
        cv_avg_f1: 0.79,
        selection_cv_mean_roc_auc: 0.86,
        cv_std_accuracy: 0.04,
        shap_sample_size: 100,
      },
      objective_value: 0.82,
      domain_task_id: 'artifact-1',
    }], task)

    expect(vm.metricKeys).toEqual(['accuracy', 'f1', 'roc_auc'])
    expect(vm.rows[0].metrics).toMatchObject({
      accuracy: 0.82,
      f1: 0.79,
      roc_auc: 0.86,
    })
  })

  it('keeps a single requested metric without falling back to internal fields', () => {
    const vm = buildComparisonVM([{
      run_id: 'one-metric',
      params: { eval_metrics: ['accuracy'] },
      metrics: { cv_avg_accuracy: 0.8, cv_std_accuracy: 0.1, shap_sample_size: 50 },
      objective_value: 0.81,
    }], task)

    expect(vm.metricKeys).toEqual(['accuracy'])
    expect(vm.rows[0].metrics).toEqual({ accuracy: 0.81 })
  })

  it('only ranks successful scored rows and gates artifact actions', () => {
    const vm = buildComparisonVM([
      {
        run_id: 'failed',
        status: 'FAILED',
        params: { model_type: 'random_forest' },
        metrics: {},
        domain_task_id: 'failed-domain-task',
      },
      ...rows,
    ], task)

    const failed = vm.rows.find(r => r.run_id === 'failed')
    const winner = vm.rows.find(r => r.run_id === 'b')
    expect(failed).toMatchObject({
      rank: null,
      is_best: false,
      can_explain: false,
      can_download: false,
      can_deploy: false,
    })
    expect(winner).toMatchObject({ rank: 1, is_best: true })
    expect(vm.rows.map(r => r.run_id)).toEqual(['b', 'a', 'failed'])
  })
})

describe('buildStrategyCardVM', () => {
  it('does not expose NaN when a strategy has no successful run', () => {
    expect(buildStrategyCardVM({ run_count: 0, full_run_count: 1, best_run: null })).toEqual({
      hasBestRun: false,
      bestRun: null,
      runCount: 0,
      fullRunCount: 1,
    })
  })
})

describe('buildFinalizationVM', () => {
  const bestRun = {
    run_id: 'winner',
    family: 'ml',
    final_test_value: null,
  }

  it('allows an idle ML task to confirm its winner', () => {
    const vm = buildFinalizationVM({
      objective_metric: 'accuracy',
      final_evaluation: { state: 'OPEN', version: 1 },
      run_stats: { running: 0 },
    }, bestRun)

    expect(vm.state).toBe('OPEN')
    expect(vm.disabled).toBe(false)
    expect(vm.actionLabel).toBe('确认最终模型')
  })

  it('blocks confirmation while runs are active', () => {
    const vm = buildFinalizationVM({
      final_evaluation: { state: 'OPEN', version: 1 },
      run_stats: { running: 2 },
    }, bestRun)

    expect(vm.disabled).toBe(true)
    expect(vm.reason).toContain('运行结束')
  })

  it('allows DL winners (B1) and blocks unknown families', () => {
    const dlVm = buildFinalizationVM({
      final_evaluation: { state: 'OPEN', version: 1 },
      run_stats: { running: 0 },
    }, { ...bestRun, family: 'dl' })
    expect(dlVm.disabled).toBe(false)
    expect(dlVm.reason).toBeNull()

    const unknownVm = buildFinalizationVM({
      final_evaluation: { state: 'OPEN', version: 1 },
      run_stats: { running: 0 },
    }, { ...bestRun, family: null })
    expect(unknownVm.disabled).toBe(true)
    expect(unknownVm.reason).toContain('模型族')
  })

  it('represents an in-progress claim', () => {
    const vm = buildFinalizationVM({
      final_evaluation: { state: 'EVALUATING', version: 1 },
    }, bestRun)

    expect(vm.disabled).toBe(true)
    expect(vm.actionLabel).toBe('正在确认')
  })

  it('exposes the finalized metric', () => {
    const vm = buildFinalizationVM({
      objective_metric: 'accuracy',
      final_evaluation: {
        state: 'FINALIZED',
        version: 1,
        winner_run_id: 'winner',
        final_metrics: { final_test_accuracy: 0.87 },
      },
    }, bestRun)

    expect(vm.disabled).toBe(true)
    expect(vm.finalValue).toBe(0.87)
    expect(vm.actionLabel).toBe('已确认最终模型')
  })

  it('allows retry after a failed claim', () => {
    const vm = buildFinalizationVM({
      final_evaluation: {
        state: 'FAILED',
        version: 1,
        error: 'artifact read failed',
      },
      run_stats: { running: 0 },
    }, bestRun)

    expect(vm.disabled).toBe(false)
    expect(vm.actionLabel).toBe('重试最终确认')
    expect(vm.error).toBe('artifact read failed')
  })

  it('refreshes task state when finalization fails', async () => {
    const events = []
    const failure = new Error('finalization failed')

    await expect(finalizeTaskAndRefresh(
      async () => { events.push('finalize'); throw failure },
      async () => { events.push('refresh') },
    )).rejects.toThrow('finalization failed')

    expect(events).toEqual(['finalize', 'refresh'])
  })
})
