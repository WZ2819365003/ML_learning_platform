import React, { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Alert, Button, Card, Empty, Progress, Space, Spin, Tag, Tooltip,
  Tree, Typography,
} from 'antd'
import {
  AppstoreOutlined, CheckCircleOutlined, ClockCircleOutlined,
  CloseCircleOutlined, ExperimentOutlined, QuestionCircleOutlined,
  ReloadOutlined, SyncOutlined, ThunderboltOutlined,
} from '@ant-design/icons'
import { modelingTaskApi } from '../../services/api'

const { Text } = Typography

/**
 * ProgressTree — three-level orchestration view for a ModelingTask.
 *
 *   ModelingTask (root)
 *     └── PlatformExperiment (strategy_type)
 *           └── ExperimentRun (one leaf per selected model / per trial)
 *
 * Reads `/api/v3/tasks/{id}/progress-tree`.  While the backend reports
 * `has_active_runs=true`, we re-fetch every 3 seconds so per-epoch DL
 * updates and ML fold completions surface without a reload.
 *
 * Leaf rendering:
 *   ML: <ExperimentOutlined/> + blue tag
 *   DL: <ThunderboltOutlined/> + purple tag
 *   unknown: <QuestionCircleOutlined/>
 */
export default function ProgressTree({ modelingTaskId, autoRefresh = true, pollMs = 3000 }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)

  const load = useCallback(async () => {
    if (!modelingTaskId) return
    setLoading(true)
    try {
      const tree = await modelingTaskApi.progressTree(modelingTaskId)
      setData(tree)
    } catch (err) {
      // Swallow — the endpoint may not be ready on older backends, and a
      // stale tree is harmless.  The caller Card still renders.
      console.warn('加载进度树失败', err)
    } finally {
      setLoading(false)
    }
  }, [modelingTaskId])

  useEffect(() => { load() }, [load])

  useEffect(() => {
    if (!autoRefresh) return
    if (!data?.has_active_runs) return
    const t = setInterval(load, pollMs)
    return () => clearInterval(t)
  }, [autoRefresh, pollMs, data?.has_active_runs, load])

  const treeData = useMemo(() => _buildTreeNodes(data), [data])
  const expandedKeys = useMemo(() => _allNodeKeys(treeData), [treeData])

  return (
    <Card
      size="small"
      title={
        <Space>
          <AppstoreOutlined />
          <span>编排进度</span>
          {data?.modeling_task && (
            <Tag color="blue">
              {data.modeling_task.experiment_count} 个批次 · {data.modeling_task.run_count} 个 Run
            </Tag>
          )}
        </Space>
      }
      extra={
        <Space>
          {data?.has_active_runs && (
            <Tag icon={<SyncOutlined spin />} color="processing">自动刷新中</Tag>
          )}
          <Button size="small" icon={<ReloadOutlined />} onClick={load} loading={loading}>
            刷新
          </Button>
        </Space>
      }
    >
      {data?.modeling_task && (
        <div style={{ marginBottom: 12 }}>
          <Space align="center" style={{ width: '100%' }}>
            <Text strong>整体进度</Text>
            <div style={{ flex: 1, minWidth: 240 }}>
              <Progress
                percent={Math.round((data.modeling_task.progress_aggregated ?? 0) * 100)}
                size="small"
                status={data.has_active_runs ? 'active' : 'normal'}
              />
            </div>
          </Space>
        </div>
      )}

      <Spin spinning={loading && !data}>
        {treeData.length === 0 ? (
          <Empty description="暂无实验批次 — 创建实验后会在这里看到进度" />
        ) : (
          <Tree
            treeData={treeData}
            expandedKeys={expandedKeys}
            selectable={false}
            showLine
            blockNode
          />
        )}
      </Spin>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Tree node construction
// ---------------------------------------------------------------------------

function _buildTreeNodes(data) {
  if (!data?.experiments) return []
  return data.experiments.map((exp) => ({
    key: `exp:${exp.id}`,
    title: <ExperimentNode exp={exp} />,
    children: (exp.runs || []).map((run) => ({
      key: `run:${run.id}`,
      title: <RunNode run={run} />,
      isLeaf: true,
    })),
  }))
}

function _allNodeKeys(nodes) {
  const keys = []
  nodes.forEach((n) => {
    keys.push(n.key)
    if (n.children) n.children.forEach((c) => keys.push(c.key))
  })
  return keys
}

// ── Experiment node ────────────────────────────────────────────────────────

function ExperimentNode({ exp }) {
  const pct = Math.round((exp.progress_aggregated ?? 0) * 100)
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '4px 0' }}>
      <AppstoreOutlined style={{ color: '#64748b' }} />
      <Text strong ellipsis style={{ minWidth: 140 }}>{exp.name}</Text>
      <Tag color={_strategyColor(exp.strategy_type)}>
        {_strategyLabel(exp.strategy_type)}
      </Tag>
      <Tag color={_statusColor(exp.status)}>{exp.status}</Tag>
      <Text type="secondary" style={{ fontSize: 11 }}>
        {exp.run_count} Run
      </Text>
      <div style={{ width: 160 }}>
        <Progress percent={pct} size="small" showInfo />
      </div>
    </div>
  )
}

// ── Run node ───────────────────────────────────────────────────────────────

function RunNode({ run }) {
  const pct = Math.round((run.progress ?? 0) * 100)
  const familyChip = _familyChip(run.family)
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '3px 0' }}>
      {familyChip}
      <Text style={{ minWidth: 120 }} ellipsis>
        <Text strong>{run.model_type || '未命名'}</Text>
        {run.trial_no != null && (
          <Text type="secondary" style={{ fontSize: 11, marginLeft: 4 }}>
            #{run.trial_no}
          </Text>
        )}
      </Text>
      <Tag color={_statusColor(run.status)} icon={_statusIcon(run.status)} style={{ margin: 0 }}>
        {run.status}
      </Tag>
      {run.current_step && (
        <Tooltip title="当前步骤">
          <Text type="secondary" style={{ fontSize: 12, minWidth: 110 }}>
            {run.current_step}
          </Text>
        </Tooltip>
      )}
      <div style={{ flex: 1, minWidth: 120 }}>
        <Progress
          percent={pct}
          size="small"
          status={_progressStatus(run.status)}
          showInfo
        />
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Styling helpers
// ---------------------------------------------------------------------------

function _familyChip(family) {
  if (family === 'dl') {
    return <Tag icon={<ThunderboltOutlined />} color="purple" style={{ margin: 0 }}>DL</Tag>
  }
  if (family === 'ml') {
    return <Tag icon={<ExperimentOutlined />} color="blue" style={{ margin: 0 }}>ML</Tag>
  }
  return <Tag icon={<QuestionCircleOutlined />} color="default" style={{ margin: 0 }}>?</Tag>
}

function _statusColor(status) {
  const k = status?.toUpperCase?.()
  if (k === 'SUCCESS' || k === 'COMPLETED') return 'green'
  if (k === 'RUNNING') return 'blue'
  if (k === 'QUEUED' || k === 'PENDING') return 'gold'
  if (k === 'FAILED') return 'red'
  if (k === 'RETRY') return 'orange'
  if (k === 'CANCELLED') return 'default'
  return 'default'
}

function _statusIcon(status) {
  const k = status?.toUpperCase?.()
  if (k === 'SUCCESS' || k === 'COMPLETED') return <CheckCircleOutlined />
  if (k === 'RUNNING') return <SyncOutlined spin />
  if (k === 'QUEUED' || k === 'PENDING') return <ClockCircleOutlined />
  if (k === 'FAILED') return <CloseCircleOutlined />
  return null
}

function _progressStatus(status) {
  const k = status?.toUpperCase?.()
  if (k === 'FAILED')    return 'exception'
  if (k === 'CANCELLED') return 'exception'
  if (k === 'SUCCESS' || k === 'COMPLETED') return 'success'
  if (k === 'RUNNING')   return 'active'
  return 'normal'
}

function _strategyLabel(s) {
  return ({
    baseline:        '基线',
    grid_search:     '网格',
    bayesian_search: '贝叶斯',
  })[s] || s || '—'
}

function _strategyColor(s) {
  return ({
    baseline:        'blue',
    grid_search:     'geekblue',
    bayesian_search: 'purple',
  })[s] || 'default'
}
