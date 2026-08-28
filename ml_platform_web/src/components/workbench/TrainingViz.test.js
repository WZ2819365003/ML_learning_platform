import { describe, expect, it } from 'vitest'
import { settleVizRequest } from './TrainingViz'

describe('settleVizRequest', () => {
  it('keeps an endpoint error alongside an empty chart payload', async () => {
    await expect(
      settleVizRequest('混淆矩阵', Promise.reject({
        response: { data: { detail: 'Model file not found' } },
      })),
    ).resolves.toEqual({
      data: null,
      error: '混淆矩阵：Model file not found',
    })
  })

  it('treats a missing optional model capability as unavailable, not a page error', async () => {
    await expect(
      settleVizRequest('特征重要度', Promise.reject({
        response: { status: 400, data: { detail: 'Model does not provide feature importance' } },
      }), 'featureImportance'),
    ).resolves.toEqual({
      data: null,
      error: null,
      unavailable: '当前模型不提供原生特征重要性。SVR 等模型可使用上方 SHAP 解释，但不会生成 feature_importances_。',
    })
  })
})
