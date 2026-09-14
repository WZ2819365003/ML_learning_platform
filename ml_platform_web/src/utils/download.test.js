import { afterEach, describe, expect, it, vi } from 'vitest'
import { downloadWithAuth, errorMessageFromBlob, filenameFromContentDisposition } from './download'

function fakeResponse({ status = 200, body = 'bytes', headers = {} } = {}) {
  const lower = Object.fromEntries(Object.entries(headers).map(([k, v]) => [k.toLowerCase(), v]))
  return {
    status,
    ok: status >= 200 && status < 300,
    headers: { get: (k) => lower[k.toLowerCase()] ?? null },
    blob: async () => new Blob([body]),
  }
}

describe('filenameFromContentDisposition', () => {
  it('读 Starlette FileResponse 的普通 filename', () => {
    expect(filenameFromContentDisposition('attachment; filename="xgboost_regressor_516a973b.joblib"'))
      .toBe('xgboost_regressor_516a973b.joblib')
  })

  it('优先读 RFC 5987 的 filename*（中文文件名走这条）', () => {
    const h = `attachment; filename="fallback.csv"; filename*=utf-8''%E9%A2%84%E6%B5%8B%E7%BB%93%E6%9E%9C.csv`
    expect(filenameFromContentDisposition(h)).toBe('预测结果.csv')
  })

  it('没有文件名时返回 null', () => {
    expect(filenameFromContentDisposition(null)).toBeNull()
    expect(filenameFromContentDisposition('attachment')).toBeNull()
  })
})

describe('errorMessageFromBlob', () => {
  it('取出后端 JSON 里的 detail', async () => {
    const blob = new Blob([JSON.stringify({ detail: 'Model file not found on disk' })])
    expect(await errorMessageFromBlob(blob, 'x')).toBe('Model file not found on disk')
  })

  it('不是 JSON 时用兜底文案', async () => {
    expect(await errorMessageFromBlob(new Blob(['<html>']), '兜底')).toBe('兜底')
  })
})

describe('downloadWithAuth', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('请求头带上 Bearer token——这是修复的核心，普通链接跳转带不上它', async () => {
    const fetchImpl = vi.fn(async () => fakeResponse({ status: 401 }))
    await downloadWithAuth('/api/models/x/download', { getToken: () => 'tok', fetchImpl }).catch(() => {})
    expect(fetchImpl).toHaveBeenCalledWith('/api/models/x/download', { headers: { Authorization: 'Bearer tok' } })
  })

  it('401 时通知登出并抛出可读错误', async () => {
    const onUnauthorized = vi.fn()
    const fetchImpl = async () => fakeResponse({ status: 401 })
    await expect(downloadWithAuth('/u', { getToken: () => 't', onUnauthorized, fetchImpl }))
      .rejects.toThrow('登录已过期')
    expect(onUnauthorized).toHaveBeenCalledOnce()
  })

  it('404 时把后端 detail 透出来，而不是静默下载一段 JSON', async () => {
    const fetchImpl = async () => fakeResponse({ status: 404, body: JSON.stringify({ detail: 'Training task not found' }) })
    await expect(downloadWithAuth('/u', { getToken: () => 't', fetchImpl })).rejects.toThrow('Training task not found')
  })

  it('成功时用响应头里的文件名触发保存', async () => {
    const clicked = []
    const anchor = { style: {}, click: () => clicked.push(anchor.download), remove: () => {} }
    vi.stubGlobal('document', { createElement: () => anchor, body: { appendChild: () => {} } })
    vi.stubGlobal('URL', { createObjectURL: () => 'blob:1', revokeObjectURL: () => {} })
    const fetchImpl = async () => fakeResponse({ headers: { 'Content-Disposition': 'attachment; filename="m.joblib"' } })

    const name = await downloadWithAuth('/u', { getToken: () => 't', fetchImpl }, 'fallback.bin')
    expect(name).toBe('m.joblib')
    expect(anchor.href).toBe('blob:1')
    expect(clicked).toEqual(['m.joblib'])
  })
})
