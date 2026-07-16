/**
 * LogViewer — professional log panel for the Run Inspector.
 *
 * Features:
 *   - Live tailing via WebSocket (`/ws/logs/{domainTaskId}`) with auto-reconnect
 *   - Level filter checkboxes (INFO / WARN / ERROR / DEBUG)
 *   - Keyword search with highlighted matches
 *   - Pause / resume follow-tail mode (when paused, new entries are held
 *     back; when resumed, they flush in one frame)
 *   - Absolute vs. relative timestamp toggle
 *   - Auto-scroll to bottom when tailing, stops when user scrolls up
 *   - Download full log as .txt
 *   - Color-coded levels with monospace font
 *
 * Inputs:
 *   historical — REST-loaded entries seeded at mount (array of {level, message, extra, created_at})
 *   domainTaskId — the TrainingTask id (string); if falsy, WS is disabled
 *   isLive — whether the owning Run is still RUNNING (controls "live" indicator)
 */
import React, { useEffect, useMemo, useRef, useState, useCallback } from 'react'
import { Input, Checkbox, Button, Space, Tag, Tooltip, Empty, Typography } from 'antd'
import {
  SearchOutlined, PauseCircleOutlined, PlayCircleOutlined, DownloadOutlined,
  ClearOutlined, VerticalAlignBottomOutlined, ClockCircleOutlined,
} from '@ant-design/icons'
import { useLogStream } from '../../hooks/useLogStream'

const { Text } = Typography

const LEVELS = ['INFO', 'WARN', 'ERROR', 'DEBUG']
const LEVEL_COLOR = {
  INFO: { bg: 'rgba(59, 130, 246, 0.1)',  fg: '#2563eb', dim: '#94a3b8' },
  WARN: { bg: 'rgba(245, 158, 11, 0.12)', fg: '#d97706', dim: '#94a3b8' },
  WARNING: { bg: 'rgba(245, 158, 11, 0.12)', fg: '#d97706', dim: '#94a3b8' },
  ERROR:{ bg: 'rgba(239, 68, 68, 0.12)',  fg: '#dc2626', dim: '#94a3b8' },
  DEBUG:{ bg: 'rgba(148, 163, 184, 0.12)',fg: '#64748b', dim: '#94a3b8' },
}

function normaliseLevel(l) {
  if (!l) return 'INFO'
  const u = String(l).toUpperCase()
  if (u === 'WARNING') return 'WARN'
  return LEVELS.includes(u) ? u : 'INFO'
}

function formatAbsolute(ts) {
  if (!ts) return ''
  try { return new Date(ts).toLocaleString('zh-CN', { hour12: false, fractionalSecondDigits: 3 }) }
  catch { return '' }
}
function formatRelative(ts, now) {
  if (!ts) return ''
  try {
    const diff = now - new Date(ts).getTime()
    if (diff < 1000) return 'just now'
    if (diff < 60_000) return `${Math.floor(diff/1000)}s ago`
    if (diff < 3_600_000) return `${Math.floor(diff/60_000)}m ago`
    return `${Math.floor(diff/3_600_000)}h ago`
  } catch { return '' }
}

/** Case-insensitive highlight. Returns an array of React children. */
function highlight(text, term) {
  if (!term) return text
  const lower = text.toLowerCase()
  const needle = term.toLowerCase()
  const parts = []
  let idx = 0
  let i = lower.indexOf(needle)
  while (i >= 0) {
    if (i > idx) parts.push(text.slice(idx, i))
    parts.push(
      <mark key={i} style={{ background: '#fde68a', padding: '0 2px', borderRadius: 3 }}>
        {text.slice(i, i + term.length)}
      </mark>
    )
    idx = i + term.length
    i = lower.indexOf(needle, idx)
  }
  if (idx < text.length) parts.push(text.slice(idx))
  return parts
}

export default function LogViewer({ historical, domainTaskId, isLive }) {
  // Connect to WS. `enabled` gate avoids opening a socket when we don't
  // have a task id yet (loading state) or when caller explicitly disables.
  const { logs, connected, paused, setPaused, clear, seedHistorical } = useLogStream({
    domainTaskId: domainTaskId || null,
    enabled: !!domainTaskId,
    maxEntries: 2000,
  })

  // Seed with REST payload whenever it changes (only meaningful on first mount).
  useEffect(() => {
    if (Array.isArray(historical) && historical.length > 0) seedHistorical(historical)
  }, [historical, seedHistorical])

  // UI state
  const [search, setSearch] = useState('')
  const [enabledLevels, setEnabledLevels] = useState(new Set(LEVELS))
  const [absoluteTime, setAbsoluteTime] = useState(false)
  const [autoScroll, setAutoScroll] = useState(true)

  // 1s tick so "relative time" stays fresh
  const [nowMs, setNowMs] = useState(Date.now())
  useEffect(() => {
    const id = setInterval(() => setNowMs(Date.now()), 1000)
    return () => clearInterval(id)
  }, [])

  const filtered = useMemo(() => {
    const term = search.trim()
    return logs.filter((l) => {
      if (!enabledLevels.has(normaliseLevel(l.level))) return false
      if (term && !(l.message || '').toLowerCase().includes(term.toLowerCase())) return false
      return true
    })
  }, [logs, search, enabledLevels])

  // Auto-scroll to bottom when tailing
  const bodyRef = useRef(null)
  const userScrolledUpRef = useRef(false)

  const onBodyScroll = useCallback(() => {
    const el = bodyRef.current
    if (!el) return
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 24
    userScrolledUpRef.current = !atBottom
    if (atBottom) setAutoScroll(true)
  }, [])

  useEffect(() => {
    if (!autoScroll) return
    const el = bodyRef.current
    if (el && !userScrolledUpRef.current) {
      el.scrollTop = el.scrollHeight
    }
  }, [filtered.length, autoScroll])

  const handleJumpToBottom = () => {
    userScrolledUpRef.current = false
    setAutoScroll(true)
    const el = bodyRef.current
    if (el) el.scrollTop = el.scrollHeight
  }

  const handleDownload = () => {
    const lines = logs.map((l) => {
      const ts = l.timestamp || ''
      const lvl = (l.level || 'INFO').padEnd(5)
      const extras = l.extra
        ? ' | ' + Object.entries(l.extra).map(([k, v]) => `${k}=${v}`).join(' | ')
        : ''
      return `${ts} | ${lvl} | ${l.message}${extras}`
    })
    const blob = new Blob([lines.join('\n')], { type: 'text/plain;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${domainTaskId || 'logs'}.log`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const levelCounts = useMemo(() => {
    const counts = { INFO: 0, WARN: 0, ERROR: 0, DEBUG: 0 }
    for (const l of logs) {
      const k = normaliseLevel(l.level)
      counts[k] = (counts[k] || 0) + 1
    }
    return counts
  }, [logs])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: 520 }}>
      {/* Toolbar */}
      <div style={{
        display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center',
        padding: '6px 2px', borderBottom: '1px solid #e2e8f0', marginBottom: 6,
      }}>
        <Input
          size="small"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          prefix={<SearchOutlined />}
          placeholder="搜索日志..."
          allowClear
          style={{ width: 200 }}
        />

        <Space size={4}>
          {LEVELS.map((lvl) => (
            <Checkbox
              key={lvl}
              checked={enabledLevels.has(lvl)}
              onChange={(e) => {
                const next = new Set(enabledLevels)
                if (e.target.checked) next.add(lvl); else next.delete(lvl)
                setEnabledLevels(next)
              }}
              style={{ fontSize: 11, marginRight: 0 }}
            >
              <span style={{ color: LEVEL_COLOR[lvl]?.fg, fontWeight: 600 }}>{lvl}</span>
              <span style={{ color: '#94a3b8', marginLeft: 3 }}>({levelCounts[lvl] || 0})</span>
            </Checkbox>
          ))}
        </Space>

        <div style={{ flex: 1 }} />

        <Tooltip title={absoluteTime ? '切换到相对时间' : '切换到绝对时间'}>
          <Button size="small" icon={<ClockCircleOutlined />}
            type={absoluteTime ? 'primary' : 'default'}
            onClick={() => setAbsoluteTime((v) => !v)} />
        </Tooltip>
        <Tooltip title={paused ? '恢复跟随' : '暂停跟随'}>
          <Button size="small"
            icon={paused ? <PlayCircleOutlined /> : <PauseCircleOutlined />}
            type={paused ? 'primary' : 'default'}
            onClick={() => setPaused(!paused)} />
        </Tooltip>
        <Tooltip title="跳到底部">
          <Button size="small" icon={<VerticalAlignBottomOutlined />}
            onClick={handleJumpToBottom} />
        </Tooltip>
        <Tooltip title="清空视图(不影响服务端)">
          <Button size="small" icon={<ClearOutlined />} onClick={clear} />
        </Tooltip>
        <Tooltip title="下载日志">
          <Button size="small" icon={<DownloadOutlined />} onClick={handleDownload}
            disabled={logs.length === 0} />
        </Tooltip>

        {isLive && (
          <Tag color={connected ? 'green' : 'orange'} style={{ margin: 0 }}>
            <span style={{
              display: 'inline-block', width: 6, height: 6, borderRadius: '50%',
              background: connected ? '#10b981' : '#f59e0b', marginRight: 4,
              animation: connected ? 'pulse 1.5s infinite' : 'none',
            }} />
            {connected ? 'LIVE' : '重连中'}
          </Tag>
        )}
      </div>

      {/* Body */}
      <div
        ref={bodyRef}
        onScroll={onBodyScroll}
        style={{
          flex: 1, overflow: 'auto', background: '#0f172a',
          borderRadius: 6, padding: '6px 10px', fontFamily: 'ui-monospace, Menlo, Monaco, monospace',
          fontSize: 11.5, lineHeight: 1.55,
        }}
      >
        {filtered.length === 0 ? (
          <div style={{ padding: 40, textAlign: 'center' }}>
            <Empty
              image={Empty.PRESENTED_IMAGE_SIMPLE}
              description={<Text style={{ color: '#64748b' }}>
                {logs.length === 0
                  ? (isLive ? '等待日志…' : '该 Run 没有日志')
                  : '没有匹配的日志(检查筛选条件)'}
              </Text>}
            />
          </div>
        ) : filtered.map((l, idx) => {
          const lvl = normaliseLevel(l.level)
          const c = LEVEL_COLOR[lvl] || LEVEL_COLOR.INFO
          const ts = absoluteTime
            ? formatAbsolute(l.timestamp)
            : formatRelative(l.timestamp, nowMs)
          return (
            <div key={idx} style={{
              display: 'flex', gap: 8, alignItems: 'flex-start',
              padding: '2px 4px', borderRadius: 3,
              background: idx % 2 === 0 ? 'transparent' : 'rgba(148, 163, 184, 0.04)',
            }}>
              <span style={{ color: '#64748b', flexShrink: 0, width: absoluteTime ? 140 : 70 }}>{ts}</span>
              <span style={{
                color: c.fg, background: c.bg, padding: '0 6px', borderRadius: 3,
                flexShrink: 0, width: 50, textAlign: 'center', fontWeight: 600,
              }}>{lvl}</span>
              <span style={{ color: '#e2e8f0', wordBreak: 'break-all', flex: 1 }}>
                {highlight(l.message || '', search)}
                {l.extra && Object.keys(l.extra).length > 0 && (
                  <span style={{ color: c.dim, marginLeft: 10 }}>
                    {Object.entries(l.extra).map(([k, v]) => (
                      <span key={k} style={{ marginRight: 8 }}>
                        <span style={{ color: '#94a3b8' }}>{k}=</span>
                        <span style={{ color: '#cbd5e1' }}>{String(v)}</span>
                      </span>
                    ))}
                  </span>
                )}
              </span>
            </div>
          )
        })}
      </div>

      {/* Footer */}
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        padding: '4px 0', marginTop: 4, fontSize: 11, color: '#64748b',
      }}>
        <span>
          显示 <strong style={{ color: '#0f172a' }}>{filtered.length}</strong>
          {' / '}共 <strong style={{ color: '#0f172a' }}>{logs.length}</strong> 条
        </span>
        <span>
          {paused && <Tag color="orange" style={{ marginRight: 4 }}>已暂停</Tag>}
          {!isLive && logs.length > 0 && <Tag>历史</Tag>}
        </span>
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
    </div>
  )
}
