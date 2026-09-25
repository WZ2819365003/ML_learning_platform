import { describe, expect, it } from 'vitest'
import { deploymentNameError } from './useDeployRun'

describe('deploymentNameError', () => {
  it('requires a non-blank deployment name', () => {
    expect(deploymentNameError('')).toBe('请填写部署名称')
    expect(deploymentNameError('   ')).toBe('请填写部署名称')
    expect(deploymentNameError('production-model')).toBeNull()
  })
})
