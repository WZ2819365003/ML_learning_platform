import { describe, expect, it } from 'vitest'

import { classifyVizUnavailable } from './vizAvailability'

function apiError(status, detail) {
  return { response: { status, data: { detail } } }
}

describe('classifyVizUnavailable', () => {
  it('treats missing model feature importance as a capability boundary', () => {
    const message = classifyVizUnavailable(
      'featureImportance',
      apiError(400, 'Model does not provide feature importance'),
    )
    expect(message).toContain('不提供原生特征重要性')
    expect(message).toContain('SHAP')
  })

  it('keeps real server failures as errors', () => {
    expect(classifyVizUnavailable('featureImportance', apiError(500, 'boom'))).toBeNull()
  })
})
