import { useSyncExternalStore } from 'react'

/**
 * 全局界面偏好。
 *
 * 只放**真的有人消费**的设置——「系统设置」以前那张表单里的字段一个都没接线，
 * 保存按钮只是 setTimeout 一秒然后弹成功。这里的每一项都必须能在别处 import 到，
 * 否则不要加。
 *
 * 页签是常驻挂载的，所以改设置要立刻影响已经打开的页面，用 external store 而不是
 * 各自读 localStorage。
 */
export const STORAGE_KEY = 'ml_platform_prefs'

export const DEFAULT_PREFS = Object.freeze({
  autoRefresh: true,
  refreshSeconds: 5,
})

export const REFRESH_SECONDS_RANGE = Object.freeze({ min: 2, max: 60 })

function clampSeconds(value) {
  const n = Number(value)
  if (!Number.isFinite(n)) return DEFAULT_PREFS.refreshSeconds
  return Math.min(REFRESH_SECONDS_RANGE.max, Math.max(REFRESH_SECONDS_RANGE.min, Math.round(n)))
}

export function normalizePrefs(raw) {
  return {
    autoRefresh: raw?.autoRefresh !== false,
    refreshSeconds: clampSeconds(raw?.refreshSeconds ?? DEFAULT_PREFS.refreshSeconds),
  }
}

function readStorage() {
  if (typeof window === 'undefined') return { ...DEFAULT_PREFS }
  try {
    return normalizePrefs(JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? '{}'))
  } catch {
    return { ...DEFAULT_PREFS }
  }
}

let current = readStorage()
const listeners = new Set()

export function getPrefs() {
  return current
}

export function setPrefs(patch) {
  const next = normalizePrefs({ ...current, ...patch })
  if (next.autoRefresh === current.autoRefresh && next.refreshSeconds === current.refreshSeconds) {
    return current
  }
  current = next
  if (typeof window !== 'undefined') {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(current))
    } catch {
      // 隐私模式下写不进去也不影响本次会话使用。
    }
  }
  listeners.forEach(listener => listener(current))
  return current
}

export function resetPrefs() {
  return setPrefs({ ...DEFAULT_PREFS })
}

function subscribe(listener) {
  listeners.add(listener)
  return () => listeners.delete(listener)
}

export function useAppSettings() {
  return useSyncExternalStore(subscribe, getPrefs, getPrefs)
}

/**
 * 页面轮询用的毫秒数；关掉自动刷新时返回 null，调用方据此不建定时器。
 */
export function usePollMs() {
  const prefs = useAppSettings()
  return prefs.autoRefresh ? prefs.refreshSeconds * 1000 : null
}
