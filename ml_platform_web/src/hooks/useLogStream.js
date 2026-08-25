/**
 * useLogStream — React hook that merges historical log entries (loaded via
 * REST) with live entries streamed from `/ws/logs/{domainTaskId}`.
 *
 * Design:
 *   - `domainTaskId` is the **TrainingTask** ID, not the Run ID or the
 *     PlatformTask ID. The backend event-bus key is `logs:{training_task_id}`.
 *   - We treat the REST call as the source of truth at mount time, and the
 *     WS as an append-only stream from that moment forward.
 *   - Auto-reconnect with backoff (1s → 2s → 4s, capped at 10s) so transient
 *     network blips don't kill the stream; the component can force-reset by
 *     changing the `domainTaskId` prop.
 *   - Auth-ready: if a bearer token is present in localStorage under
 *     `ml_platform_token`, it's appended as `?token=` on the WS URL — this
 *     matches the FastAPI pattern where sub-protocols aren't easily set
 *     from the browser. If no token, we connect un-authenticated (dev mode).
 *
 * Public API:
 *   const { logs, connected, paused, setPaused, clear } = useLogStream({
 *     domainTaskId: 'abc123',
 *     enabled: true,
 *     maxEntries: 2000,
 *   })
 */
import { useEffect, useRef, useState, useCallback } from 'react'

/**
 * Where to open the log socket.
 *
 * Defaults to the page's own origin, NOT a fixed localhost. The previous
 * default (`ws://127.0.0.1:8000`) is resolved by the *browser*, so on any
 * deployment reached over the network it pointed at the viewer's own machine
 * rather than the server. Worse than failing outright: a developer running a
 * local backend on :8000 got a socket that connected happily to the wrong
 * backend, subscribed to a task id it had never heard of, and streamed
 * nothing — which reads exactly like "the logs are lagging".
 *
 * Same-origin works everywhere the app is actually served: nginx proxies
 * `/ws/` with the Upgrade headers, and the Vite dev server proxies `/ws` too.
 * VITE_WS_BASE remains an override for split-origin setups.
 */
function defaultWsBase() {
  if (typeof window === 'undefined' || !window.location?.host) {
    return 'ws://127.0.0.1:8000'   // SSR/tests — no origin to derive from
  }
  const scheme = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${scheme}//${window.location.host}`
}

const WS_BASE = (import.meta.env.VITE_WS_BASE || defaultWsBase()).replace(/\/$/, '')
const MAX_BACKOFF_MS = 10_000

export function useLogStream({ domainTaskId, enabled = true, maxEntries = 2000 } = {}) {
  const [logs, setLogs] = useState([])
  const [connected, setConnected] = useState(false)
  const [paused, setPaused] = useState(false)

  const wsRef = useRef(null)
  const backoffRef = useRef(1000)
  const reconnectTimerRef = useRef(null)
  // Mirror `paused` into a ref so the onmessage handler always reads the
  // latest value without re-creating the WS on every pause toggle.
  const pausedRef = useRef(paused)
  useEffect(() => { pausedRef.current = paused }, [paused])

  const clear = useCallback(() => setLogs([]), [])

  const appendLogs = useCallback((incoming) => {
    setLogs((prev) => {
      const merged = prev.concat(incoming)
      if (merged.length > maxEntries) return merged.slice(merged.length - maxEntries)
      return merged
    })
  }, [maxEntries])

  /**
   * Allow the consumer to seed the list with historical entries fetched
   * via REST. Called with the array returned by `/inspector`.
   */
  const seedHistorical = useCallback((entries) => {
    if (!Array.isArray(entries)) return
    const normalised = entries.map((e) => ({
      level: e.level || 'INFO',
      message: e.message,
      extra: e.extra,
      timestamp: e.created_at || e.timestamp,
      _source: 'rest',
    }))
    setLogs(normalised)
  }, [])

  useEffect(() => {
    if (!enabled || !domainTaskId) return undefined

    let disposed = false

    const connect = () => {
      if (disposed) return

      const token =
        typeof window !== 'undefined' ? window.localStorage?.getItem('ml_platform_token') : null
      const qs = token ? `?token=${encodeURIComponent(token)}` : ''
      const url = `${WS_BASE}/ws/logs/${domainTaskId}${qs}`

      let ws
      try {
        ws = new WebSocket(url)
      } catch (err) {
        scheduleReconnect()
        return
      }
      wsRef.current = ws

      ws.onopen = () => {
        if (disposed) { ws.close(); return }
        setConnected(true)
        backoffRef.current = 1000
      }

      ws.onmessage = (ev) => {
        if (pausedRef.current) return
        try {
          const data = JSON.parse(ev.data)
          if (data?.type === 'log' && data?.message != null) {
            appendLogs([{
              level: data.level || 'INFO',
              message: data.message,
              extra: data.extra,
              timestamp: data.timestamp,
              _source: 'ws',
            }])
          }
        } catch {
          // malformed payload — ignore
        }
      }

      ws.onerror = () => { /* onclose will handle reconnect */ }

      ws.onclose = () => {
        setConnected(false)
        if (!disposed) scheduleReconnect()
      }
    }

    const scheduleReconnect = () => {
      if (disposed) return
      const delay = Math.min(backoffRef.current, MAX_BACKOFF_MS)
      backoffRef.current = Math.min(backoffRef.current * 2, MAX_BACKOFF_MS)
      reconnectTimerRef.current = setTimeout(connect, delay)
    }

    connect()

    return () => {
      disposed = true
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current)
      if (wsRef.current) {
        try { wsRef.current.close() } catch { /* ignore */ }
      }
      setConnected(false)
    }
  }, [domainTaskId, enabled, appendLogs])

  return { logs, connected, paused, setPaused, clear, seedHistorical }
}
