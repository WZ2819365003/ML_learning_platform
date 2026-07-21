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
})
