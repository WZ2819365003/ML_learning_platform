import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  DEFAULT_PREFS,
  REFRESH_SECONDS_RANGE,
  getPrefs,
  normalizePrefs,
  resetPrefs,
  setPrefs,
} from './useAppSettings'

// vitest 跑在 node 环境，没有 window；模块本身对此有防御，这里补一个最小实现来
// 覆盖落盘分支。
function stubStorage() {
  const map = new Map()
  return {
    getItem: key => (map.has(key) ? map.get(key) : null),
    setItem: (key, value) => { map.set(key, String(value)) },
    removeItem: key => { map.delete(key) },
  }
}

beforeEach(() => {
  globalThis.window = { localStorage: stubStorage() }
  resetPrefs()
})

afterEach(() => {
  delete globalThis.window
})

describe('normalizePrefs', () => {
  it('缺字段时回落到默认值', () => {
    expect(normalizePrefs({})).toEqual({ autoRefresh: true, refreshSeconds: 5 })
    expect(normalizePrefs(null)).toEqual(DEFAULT_PREFS)
  })

  it('把间隔夹到合法区间，避免 0 秒轮询打爆后端', () => {
    expect(normalizePrefs({ refreshSeconds: 0 }).refreshSeconds).toBe(REFRESH_SECONDS_RANGE.min)
    expect(normalizePrefs({ refreshSeconds: -5 }).refreshSeconds).toBe(REFRESH_SECONDS_RANGE.min)
    expect(normalizePrefs({ refreshSeconds: 9999 }).refreshSeconds).toBe(REFRESH_SECONDS_RANGE.max)
    expect(normalizePrefs({ refreshSeconds: '12' }).refreshSeconds).toBe(12)
    expect(normalizePrefs({ refreshSeconds: 'abc' }).refreshSeconds).toBe(DEFAULT_PREFS.refreshSeconds)
  })

  it('只有显式 false 才关闭自动刷新', () => {
    expect(normalizePrefs({ autoRefresh: false }).autoRefresh).toBe(false)
    expect(normalizePrefs({ autoRefresh: undefined }).autoRefresh).toBe(true)
  })
})

describe('setPrefs', () => {
  it('值没变时不换引用，避免订阅方空转重渲染', () => {
    const before = getPrefs()
    expect(setPrefs({ refreshSeconds: before.refreshSeconds })).toBe(before)
  })

  it('值变了就换引用并落盘', () => {
    const before = getPrefs()
    const after = setPrefs({ refreshSeconds: 20 })
    expect(after).not.toBe(before)
    expect(after.refreshSeconds).toBe(20)
    expect(JSON.parse(window.localStorage.getItem('ml_platform_prefs')).refreshSeconds).toBe(20)
  })

  it('写入越界值会被夹住', () => {
    expect(setPrefs({ refreshSeconds: 500 }).refreshSeconds).toBe(REFRESH_SECONDS_RANGE.max)
  })

  it('localStorage 写失败不影响本次会话', () => {
    const spy = vi.spyOn(window.localStorage, 'setItem').mockImplementation(() => {
      throw new Error('QuotaExceeded')
    })
    expect(() => setPrefs({ refreshSeconds: 7 })).not.toThrow()
    expect(getPrefs().refreshSeconds).toBe(7)
    spy.mockRestore()
  })
})
