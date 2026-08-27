import React, { useMemo, useState } from 'react'
import {
  Card, Table, Tag, Button, Space, Tooltip, Typography, Empty, Alert, Spin,
  Modal, Input, message, Row, Col,
} from 'antd'
import {
  TrophyOutlined, BulbOutlined, DownloadOutlined, CloudUploadOutlined,
  ReloadOutlined, LineChartOutlined, SafetyCertificateOutlined,
} from '@ant-design/icons'
import {
  buildComparisonVM,
  buildFinalizationVM,
  finalizeTaskAndRefresh,
} from '../../utils/comparison'
import { useDeployRun } from '../../hooks/useDeployRun'
import { modelingTaskApi, runModelDownloadUrl } from '../../services/api'
import RunInspector from './RunInspector'

const { Text } = Typography

/**
 * ModelComparison — model-first comparison view shared by the workflow 训练过程
 * step and the task detail page. Consumes raw leaderboard/runs rows via
 * buildComparisonVM; renders a dynamic-metric leaderboard, a per-model bar
 * chart, an embedded strategy comparison, per-run drill-down, and deploy.
 *
 * props: task, rows (raw), loading, error, onRefresh
 */

/**
 * Per-metric best value, so a column can mark its winner.
 *
 * Direction is per *task*, not per metric: a task optimising rmse wants the
 * minimum everywhere, and one optimising accuracy the maximum. That is a
 * simplification — a task could in principle mix rmse and r2 — but it matches
 * how the leaderboard already ranks, so the highlight can never disagree with
 * the 排名 column beside it.
 */
function bestPerMetric(rows, metricKeys, direction) {
  const best = {}
  for (const key of metricKeys) {
    const values = rows
      .map((r) => r.metrics?.[key])
      .filter((v) => typeof v === 'number')
    if (values.length === 0) continue
    best[key] = direction === 'min' ? Math.min(...values) : Math.max(...values)
  }
  return best
}

/** Gap to the champion on the objective metric, signed so worse is always positive. */
function objectiveDelta(row, championValue, direction) {
  const v = row.objective_value
  if (typeof v !== 'number' || typeof championValue !== 'number') return null
  return direction === 'min' ? v - championValue : championValue - v
}

/**
 * The tuning surface of a run — the values that actually differed between
 * trials, not the plumbing.
 *
 * `ExperimentRun.params` mixes both: the real hyperparameters live nested
 * under `hyperparameters`, while the top level carries dataset_id, family,
 * target_column and friends. Rendering the top level verbatim showed a UUID
 * next to `n_estimators` and buried the interesting values.
 *
 * DL runs nest one level further (arch_config / opt_config / train_config), so
 * a single object value is flattened into `block.key` entries rather than
 * dumped as JSON.
 */
function tuningParams(params) {
  const source = params?.hyperparameters && typeof params.hyperparameters === 'object'
    ? params.hyperparameters
    : Object.fromEntries(
        Object.entries(params || {}).filter(([k]) => !PLUMBING_PARAM_KEYS.has(k)),
      )

  const flat = []
  for (const [key, value] of Object.entries(source)) {
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      for (const [inner, innerValue] of Object.entries(value)) {
        flat.push([`${key.replace(/_config$/, '')}.${inner}`, innerValue])
      }
    } else {
      flat.push([key, value])
    }
  }
  return flat
}

const PLUMBING_PARAM_KEYS = new Set([
  'model_type', 'family', 'task_type', 'dataset_id', 'target_column', 'eval_metrics',
])

/** Hyperparameters + every metric the run produced, for the expanded row. */
function RowDetail({ row }) {
  const params = tuningParams(row.params)
  const metrics = Object.entries(row.all_metrics || {})
    .filter(([, v]) => typeof v === 'number')
  return (
    <Row gutter={[16, 8]} style={{ padding: '4px 8px 8px' }}>
      <Col xs={24} md={12}>
        <Text strong style={{ fontSize: 12 }}>超参数</Text>
        {params.length === 0 ? (
          <div><Text type="secondary" style={{ fontSize: 12 }}>该 Run 使用注册表默认值，未覆盖任何超参</Text></div>
        ) : (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 6 }}>
            {params.map(([k, v]) => (
              <Tag key={k} style={{ margin: 0, fontFamily: 'monospace', fontSize: 11 }}>
                {k} = {typeof v === 'object' ? JSON.stringify(v) : String(v)}
              </Tag>
            ))}
          </div>
        )}
      </Col>
      <Col xs={24} md={12}>
        <Text strong style={{ fontSize: 12 }}>全部指标</Text>
        {metrics.length === 0 ? (
          <div><Text type="secondary" style={{ fontSize: 12 }}>无</Text></div>
        ) : (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 6 }}>
            {metrics.map(([k, v]) => (
              <Tag key={k} style={{ margin: 0, fontFamily: 'monospace', fontSize: 11 }}>
                {k} = {v.toFixed(4)}
              </Tag>
            ))}
          </div>
        )}
      </Col>
    </Row>
  )
}

export default function ModelComparison({
  task, rows = [], loading = false, error = null, onRefresh,
  fillHeight = false,   // stretch to the host's fixed box instead of to the rows
}) {
  const vm = useMemo(() => buildComparisonVM(rows, task), [rows, task])
  const [inspectorRunId, setInspectorRunId] = useState(null)
  const [inspectorTab, setInspectorTab] = useState('overview')
  const [deployModal, setDeployModal] = useState(null) // { runId, name }
  const [finalizing, setFinalizing] = useState(false)
  const { deploying, error: deployError, deploy, reset: resetDeploy } = useDeployRun(task)

  const openInspector = (rid, tab = 'overview') => { setInspectorRunId(rid); setInspectorTab(tab) }
  const bestRun = vm.rows.find(r => r.is_best)
  const finalization = useMemo(
    () => buildFinalizationVM(task, bestRun),
    [task, bestRun],
  )

  const finalizeWinner = () => {
    Modal.confirm({
      title: '确认最终模型',
      icon: <SafetyCertificateOutlined />,
      content: '确认后将使用封存测试集评估当前冠军，并停止该任务继续新增实验批次。',
      okText: '确认并评估',
      cancelText: '取消',
      onOk: async () => {
        setFinalizing(true)
        try {
          await finalizeTaskAndRefresh(
            () => modelingTaskApi.finalize(task.id),
            onRefresh,
          )
          message.success('最终模型已确认')
        } catch (err) {
          message.error(err?.response?.data?.detail || '最终确认失败')
          throw err
        } finally {
          setFinalizing(false)
        }
      },
    })
  }

  // Per-model aggregate: best trial value per model_type for the active metric.

  if (loading) {
    return <Card size="small"><Spin tip="加载对比数据…"><div style={{ height: 120 }} /></Spin></Card>
  }
  if (error) {
    return <Alert type="error" showIcon message="加载对比失败" description={error} />
  }
  if (!vm.rows.length) {
    return <Card size="small"><Empty description="还没有 Run — 先在「训练过程」启动一批模型" /></Card>
  }

  const metricBest = bestPerMetric(vm.rows, vm.metricKeys, vm.objective_direction)
  const championValue = bestRun?.objective_value ?? null

  const columns = [
    { title: '排名', key: 'rank', width: 64,
      render: (_, r) => r.is_best
        ? <Tag color="gold" icon={<TrophyOutlined />}>1</Tag>
        : <span>{r.rank ?? '-'}</span> },
    { title: '模型', key: 'model',
      render: (_, r) => <Space size={4}><Tag>{r.model_type}</Tag>{r.family && <Tag style={{ fontSize: 10, margin: 0 }}>{r.family}</Tag>}</Space> },
    { title: '策略', dataIndex: 'strategy_type', key: 'strategy_type', width: 110, render: s => <Tag color="blue">{s}</Tag> },
    { title: '状态', dataIndex: 'status', key: 'status', width: 90,
      render: s => { const u = String(s).toUpperCase(); return <Tag color={u === 'SUCCESS' ? 'success' : u === 'FAILED' ? 'error' : 'processing'}>{u}</Tag> } },
    ...vm.metricKeys.map(k => ({
      title: k === vm.objective_metric ? <b>{k}（选择分）</b> : k, key: `m_${k}`, width: 128,
      sorter: (a, b) => (a.metrics[k] ?? -Infinity) - (b.metrics[k] ?? -Infinity),
      render: (_, r) => {
        const v = r.metrics?.[k]
        if (typeof v !== 'number') return <span style={{ color: '#cbd5e1' }}>-</span>
        // Winner of this column gets a tinted chip, so the eye can scan a
        // column instead of comparing four-decimal numbers by hand.
        const isColumnBest = metricBest[k] != null && v === metricBest[k]
        const isObjective = k === vm.objective_metric
        return (
          <code style={{
            color: isColumnBest ? '#047857' : (isObjective ? '#2563eb' : '#334155'),
            fontWeight: isColumnBest || isObjective ? 600 : 400,
            background: isColumnBest ? 'rgba(16,185,129,0.10)' : undefined,
            padding: isColumnBest ? '1px 6px' : undefined,
            borderRadius: isColumnBest ? 4 : undefined,
          }}>{v.toFixed(4)}</code>
        )
      },
    })),
    // How far behind the champion — the number people actually want when
    // deciding whether the winner is meaningfully better or a rounding win.
    { title: '差距', key: 'delta', width: 96,
      render: (_, r) => {
        const d = objectiveDelta(r, championValue, vm.objective_direction)
        if (d == null) return <span style={{ color: '#cbd5e1' }}>-</span>
        if (Math.abs(d) < 1e-12) return <Tag color="gold" style={{ margin: 0 }}>冠军</Tag>
        return <Text type="secondary" style={{ fontFamily: 'monospace', fontSize: 12 }}>+{d.toFixed(4)}</Text>
      } },
    { title: '操作', key: 'actions', width: 220,
      render: (_, r) => (
        <Space size={2}>
          <Button size="small" type="link" onClick={() => openInspector(r.run_id, 'overview')}>详情</Button>
          <Tooltip title={r.can_explain ? '解释此模型' : '仅成功且有产物的 Run 可解释'}>
            <Button size="small" type="link" icon={<BulbOutlined />} disabled={!r.can_explain}
              onClick={() => openInspector(r.run_id, 'shap')}>解释</Button>
          </Tooltip>
          {r.can_download && (
            <Tooltip title="下载模型"><Button size="small" type="link" icon={<DownloadOutlined />}
              href={runModelDownloadUrl(r.domain_task_id)} target="_blank" /></Tooltip>
          )}
          <Tooltip title={r.can_deploy ? '部署此模型' : '仅成功且有产物的 Run 可部署'}>
            <Button size="small" type="link" icon={<CloudUploadOutlined />} disabled={!r.can_deploy}
              onClick={() => { resetDeploy(); setDeployModal({ runId: r.run_id, name: `${task.name}-${r.model_type}` }) }}>部署</Button>
          </Tooltip>
        </Space>
      ) },
  ]

  return (
    <Space direction="vertical" size={12} style={fillHeight
      ? { width: '100%', height: '100%', display: 'flex' }
      : { width: '100%' }}>
      {bestRun && (
        <Card size="small" bodyStyle={{ padding: '10px 16px' }}>
          <Space wrap size={12}>
            <TrophyOutlined style={{ color: '#f59e0b' }} />
            <Text strong>最优模型：{bestRun.model_type}</Text>
            <Text type="secondary">选择分 {vm.objective_metric} = <code style={{ color: '#10b981' }}>{bestRun.objective_value?.toFixed(4)}</code></Text>
            {finalization.state === 'FINALIZED' ? (
              <>
                <Tag color="success" icon={<SafetyCertificateOutlined />}>已确认最终模型</Tag>
                {typeof finalization.finalValue === 'number' && (
                  <Text type="secondary">最终测试 {vm.objective_metric} = <code>{finalization.finalValue.toFixed(4)}</code></Text>
                )}
              </>
            ) : (
              <Tooltip title={finalization.reason || '在封存测试集上确认当前冠军'}>
                <Button size="small" icon={<SafetyCertificateOutlined />}
                  loading={finalizing || finalization.state === 'EVALUATING'}
                  disabled={finalization.disabled}
                  onClick={finalizeWinner}>{finalization.actionLabel}</Button>
              </Tooltip>
            )}
            <Button size="small" type="primary" icon={<CloudUploadOutlined />} loading={deploying} disabled={!bestRun.can_deploy}
              onClick={() => { resetDeploy(); setDeployModal({ runId: bestRun.run_id, name: `${task.name}-${bestRun.model_type}` }) }}>部署最优</Button>
          </Space>
          {finalization.error && (
            <Alert type="error" showIcon message={finalization.error} style={{ marginTop: 10 }} />
          )}
        </Card>
      )}

      {/* In fillHeight mode this card absorbs whatever height the 最优模型
          bar above it leaves, so the frame has no dead band at the bottom. */}
      <Card size="small" title={<span><LineChartOutlined /> 模型对比（{vm.rows.length}）</span>}
        style={fillHeight ? { flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' } : undefined}
        styles={{
          body: fillHeight
            ? { padding: 0, flex: 1, minHeight: 0, overflowY: 'auto' }
            : { padding: 0 },
        }}
        extra={onRefresh && <Button size="small" icon={<ReloadOutlined />} onClick={onRefresh}>刷新</Button>}>
        <Table size="small" rowKey="run_id" columns={columns} dataSource={vm.rows} scroll={{ x: 940 }}
          expandable={{
            expandedRowRender: (r) => <RowDetail row={r} />,
            rowExpandable: (r) =>
              Object.keys(r.params || {}).length > 0 || Object.keys(r.all_metrics || {}).length > 0,
          }}
          pagination={vm.rows.length > 10 ? { pageSize: 10 } : false} />
      </Card>

      {/* The metric bar chart is gone: it plotted one bar per model from the
          same numbers the table above already shows, sorted the same way.
          按策略对比 moved out to its own tab — it is a different question
          (which *strategy* pays off) from "which run won". */}

      <Modal open={!!deployModal} title="部署此模型" okText="部署" cancelText="取消" confirmLoading={deploying}
        onCancel={() => { setDeployModal(null); resetDeploy() }}
        onOk={async () => {
          const result = await deploy(deployModal.runId, { name: deployModal.name })
          if (result) setDeployModal(null)
        }}>
        <Text type="secondary" style={{ fontSize: 12 }}>部署名称</Text>
        <Input value={deployModal?.name || ''} onChange={e => setDeployModal(m => ({ ...m, name: e.target.value }))} style={{ marginTop: 6 }} />
        {deployError && <Alert type="error" showIcon message={deployError} style={{ marginTop: 10 }} />}
      </Modal>

      <RunInspector open={!!inspectorRunId} runId={inspectorRunId} defaultTab={inspectorTab}
        onClose={() => setInspectorRunId(null)} />
    </Space>
  )
}
