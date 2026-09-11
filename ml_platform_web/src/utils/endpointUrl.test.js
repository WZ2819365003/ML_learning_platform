import { describe, expect, it } from 'vitest'
import { absoluteEndpoint } from './endpointUrl'

describe('absoluteEndpoint', () => {
  const origin = 'http://203.176.93.249:18081'

  it('把网关相对路径拼到当前 host 上', () => {
    expect(absoluteEndpoint('/api/ts/deployments/abc/predict', origin))
      .toBe('http://203.176.93.249:18081/api/ts/deployments/abc/predict')
  })

  it('ML 推理路径不带 /api 前缀，照样能拼', () => {
    expect(absoluteEndpoint('/inference/abc/predict', origin))
      .toBe('http://203.176.93.249:18081/inference/abc/predict')
  })

  it('已经是绝对地址的原样返回（老数据里存过 127.0.0.1）', () => {
    const legacy = 'http://127.0.0.1:8000/api/ts/deployments/abc/predict'
    expect(absoluteEndpoint(legacy, origin)).toBe(legacy)
  })

  it('不重复斜杠', () => {
    expect(absoluteEndpoint('/x', 'http://h/')).toBe('http://h/x')
    expect(absoluteEndpoint('x', 'http://h')).toBe('http://h/x')
  })

  it('空值返回空串', () => {
    expect(absoluteEndpoint(null, origin)).toBe('')
    expect(absoluteEndpoint(undefined, origin)).toBe('')
    expect(absoluteEndpoint('', origin)).toBe('')
  })
})
