import { describe, expect, it } from 'vitest'
import { budgetStarvation, plannedTrialsPerModel } from './trialBudget'

// 与 registry/tuning_spaces.yaml 里回归三件套一致
const SPACES = {
  xgboost_regressor: {
    grid_values: { n_estimators: [200, 400], learning_rate: [0.03, 0.1], max_depth: [4, 6, 8] },
    distribution: { n_estimators: { type: 'int', low: 100, high: 800 } },
  },
  lightgbm_regressor: {
    grid_values: { n_estimators: [200, 400], learning_rate: [0.03, 0.1], num_leaves: [31, 63] },
    distribution: { num_leaves: { type: 'int', low: 16, high: 256 } },
  },
  random_forest_regressor: {
    grid_values: { n_estimators: [100, 200, 400], max_depth: [null, 6, 12] },
    distribution: { max_depth: { type: 'int', low: 3, high: 24 } },
  },
  mlp_dl: { family: 'dl' },
}
const MODELS = ['xgboost_regressor', 'lightgbm_regressor', 'random_forest_regressor']

describe('plannedTrialsPerModel', () => {
  it('网格：没改过网格时用注册表组合数', () => {
    expect(plannedTrialsPerModel('grid_search', MODELS, SPACES, {}, 10)).toEqual([
      { model: 'xgboost_regressor', planned: 12 },
      { model: 'lightgbm_regressor', planned: 8 },
      { model: 'random_forest_regressor', planned: 9 },
    ])
  })

  it('网格：用户改过的网格整体替换注册表（和后端一致，不合并）', () => {
    const params = { grid: { xgboost_regressor: { max_depth: [3, 5] } } }
    expect(plannedTrialsPerModel('grid_search', ['xgboost_regressor'], SPACES, params, 10))
      .toEqual([{ model: 'xgboost_regressor', planned: 2 }])
  })

  it('贝叶斯：每个有分布的模型都是 n_trials_per_model 次', () => {
    expect(plannedTrialsPerModel('bayesian_search', MODELS, SPACES, {}, 20).map(p => p.planned))
      .toEqual([20, 20, 20])
  })

  it('深度学习模型不参与网格/贝叶斯展开', () => {
    expect(plannedTrialsPerModel('grid_search', ['mlp_dl'], SPACES, {}, 10)).toEqual([])
  })
})

describe('budgetStarvation', () => {
  it('复现线上事故：3×20、上限 20 → 后两个模型 0 次', () => {
    const planned = plannedTrialsPerModel('bayesian_search', MODELS, SPACES, {}, 20)
    const { total, starved } = budgetStarvation(planned, 20)
    expect(total).toBe(60)
    expect(starved).toEqual([
      { model: 'lightgbm_regressor', planned: 20, got: 0 },
      { model: 'random_forest_regressor', planned: 20, got: 0 },
    ])
  })

  it('部分截断也算：上限 25 时随机森林 0、lightgbm 只剩 5', () => {
    const planned = plannedTrialsPerModel('bayesian_search', MODELS, SPACES, {}, 20)
    expect(budgetStarvation(planned, 25).starved).toEqual([
      { model: 'lightgbm_regressor', planned: 20, got: 5 },
      { model: 'random_forest_regressor', planned: 20, got: 0 },
    ])
  })

  it('上限足够或未设置时没问题', () => {
    const planned = plannedTrialsPerModel('grid_search', MODELS, SPACES, {}, 10)
    expect(budgetStarvation(planned, 29).starved).toEqual([])
    expect(budgetStarvation(planned, null).starved).toEqual([])
  })

  it('只有一个模型时截断是安全阀，不报', () => {
    const planned = plannedTrialsPerModel('grid_search', ['xgboost_regressor'], SPACES, {}, 10)
    expect(budgetStarvation(planned, 5).starved).toEqual([])
  })
})
