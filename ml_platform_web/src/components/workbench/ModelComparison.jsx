import React, { useMemo, useState } from 'react'
import {
  Card, Table, Tag, Button, Space, Tooltip, Typography, Empty, Alert, Spin,
  Select, Modal, Input,
} from 'antd'
import {
  TrophyOutlined, BulbOutlined, DownloadOutlined, CloudUploadOutlined,
  ReloadOutlined, LineChartOutlined,
} from '@ant-design/icons'
import { buildComparisonVM } from '../../utils/comparison'
import { useDeployRun } from '../../hooks/useDeployRun'
import { runModelDownloadUrl } from '../../services/api'
import EChart from '../EChart'
import RunInspector from './RunInspector'
import StrategyCompareTab from './StrategyCompareTab'

const { Text } = Typography

const canDeploy = (r) =>
  String(r.status).toUpperCase() === 'SUCCESS' && !!r.domain_task_id

/**
 * ModelComparison — model-first comparison view shared by the workflow 训练过程
 * step and the task detail page. Consumes raw leaderboard/runs rows via
 * buildComparisonVM; renders a dynamic-metric leaderboard, a per-model bar
 * chart, an embedded strategy comparison, per-run drill-down, and deploy.
 *
 * props: task, rows (raw), loading, error, onRefresh
 */
export default function ModelComparison({ task, rows = [], loading = false, error = null, onRefresh }) {
  const vm = useMemo(() => buildComparisonVM(rows, task), [rows, task])
  const [inspectorRunId, setInspectorRunId] = useState(null)
  const [inspectorTab, setInspectorTab] = useState('overview')
  const [chartMetric, setChartMetric] = useState(null)
  const [deployModal, setDeployModal] = useState(null) // { runId, name }
  const { deploying, deploy } = useDeployRun(task)

  const openInspector = (rid, tab = 'overview') => { setInspectorRunId(rid); setInspectorTab(tab) }
  const bestRun = vm.rows.find(r => r.is_best)
  const activeMetric = chartMetric && vm.metricKeys.includes(chartMetric) ? chartMetric : vm.objective_metric

  // Per-model aggregate: best trial value per model_type for the active metric.
  const chartData = useMemo(() => {
    const byModel = {}
    vm.rows.forEach(r => {
      const v = r.metrics?.[activeMetric]
      if (typeof v !== 'number') return
      byModel[r.model_type] = r.model_type in byModel
        ? (vm.objective_direction === 'min' ? Math.min(byModel[r.model_type], v) : Math.max(byModel[r.model_type], v))
        : v
    })
    const entries = Object.entries(byModel)
      .sort((a, b) => vm.objective_direction === 'min' ? a[1] - b[1] : b[1] - a[1])
    return { models: entries.map(e => e[0]), values: entries.map(e => e[1]) }
  }, [vm, activeMetric])

  const barOption = {
    grid: { left: 56, right: 16, top: 16, bottom: 64 },
    xAxis: { type: 'category', data: chartData.models, axisLabel: { rotate: 20, fontSize: 11 } },
    yAxis: { type: 'value' },
    tooltip: { trigger: 'axis' },
    series: [{
      type: 'bar', data: chartData.values, itemStyle: { color: '#2563eb', borderRadius: [4, 4, 0, 0] },
      label: { show: true, position: 'top', fontSize: 10, formatter: p => (typeof p.value === 'number' ? p.value.toFixed(4) : '') },
    }],
  }

  if (loading) {
    return <Card size="small"><Spin tip="加载对比数据…"><div style={{ height: 120 }} /></Spin></Card>
  }
  if (error) {
    return <Alert type="error" showIcon message="加载对比失败" description={error} />
  }
  if (!vm.rows.length) {
    return <Card size="small"><Empty description="还没有 Run — 先在「训练过程」启动一批模型" /></Card>
  }

  const columns = [
    { title: '排名', key: 'rank', width: 64,
      render: (_, r, i) => r.is_best ? <Tag color="gold" icon={<TrophyOutlined />}>1</Tag> : <span>{i + 1}</span> },
    { title: '模型', key: 'model',
      render: (_, r) => <Space size={4}><Tag>{r.model_type}</Tag>{r.family && <Tag style={{ fontSize: 10, margin: 0 }}>{r.family}</Tag>}</Space> },
    { title: '策略', dataIndex: 'strategy_type', key: 'strategy_type', width: 110, render: s => <Tag color="blue">{s}</Tag> },
    { title: '状态', dataIndex: 'status', key: 'status', width: 90,
      render: s => { const u = String(s).toUpperCase(); return <Tag color={u === 'SUCCESS' ? 'success' : u === 'FAILED' ? 'error' : 'processing'}>{u}</Tag> } },
    ...vm.metricKeys.map(k => ({
      title: k === vm.objective_metric ? <b>{k}</b> : k, key: `m_${k}`, width: 108,
      sorter: (a, b) => (a.metrics[k] ?? -Infinity) - (b.metrics[k] ?? -Infinity),
      render: (_, r) => {
        const v = r.metrics?.[k]
        return typeof v === 'number'
          ? <code style={{ color: k === vm.objective_metric ? '#2563eb' : '#334155', fontWeight: k === vm.objective_metric ? 600 : 400 }}>{v.toFixed(4)}</code>
          : <span style={{ color: '#cbd5e1' }}>-</span>
      },
    })),
    { title: '操作', key: 'actions', width: 220,
      render: (_, r) => (
        <Space size={2}>
          <Button size="small" type="link" onClick={() => openInspector(r.run_id, 'overview')}>详情</Button>
          <Button size="small" type="link" icon={<BulbOutlined />} onClick={() => openInspector(r.run_id, 'shap')}>解释</Button>
          {r.domain_task_id && (
            <Tooltip title="下载模型"><Button size="small" type="link" icon={<DownloadOutlined />}
              href={runModelDownloadUrl(r.domain_task_id)} target="_blank" /></Tooltip>
          )}
          <Tooltip title={canDeploy(r) ? '部署此模型' : '仅成功且有产物的 Run 可部署'}>
            <Button size="small" type="link" icon={<CloudUploadOutlined />} disabled={!canDeploy(r)}
              onClick={() => setDeployModal({ runId: r.run_id, name: `${task.name}-${r.model_type}` })}>部署</Button>
          </Tooltip>
        </Space>
      ) },
  ]

  return (
    <Space direction="vertical" size={12} style={{ width: '100%' }}>
      {bestRun && (
        <Card size="small" bodyStyle={{ padding: '10px 16px' }}>
          <Space wrap size={12}>
            <TrophyOutlined style={{ color: '#f59e0b' }} />
            <Text strong>最优模型：{bestRun.model_type}</Text>
            <Text type="secondary">{vm.objective_metric} = <code style={{ color: '#10b981' }}>{bestRun.objective_value?.toFixed(4)}</code></Text>
            <Button size="small" type="primary" icon={<CloudUploadOutlined />} loading={deploying} disabled={!canDeploy(bestRun)}
              onClick={() => setDeployModal({ runId: bestRun.run_id, name: `${task.name}-${bestRun.model_type}` })}>部署最优</Button>
          </Space>
        </Card>
      )}

      <Card size="small" title={<span><LineChartOutlined /> 模型对比（{vm.rows.length}）</span>} bodyStyle={{ padding: 0 }}
        extra={onRefresh && <Button size="small" icon={<ReloadOutlined />} onClick={onRefresh}>刷新</Button>}>
        <Table size="small" rowKey="run_id" columns={columns} dataSource={vm.rows} scroll={{ x: 820 }}
          pagination={vm.rows.length > 10 ? { pageSize: 10 } : false} />
      </Card>

      <Card size="small" title="指标柱状对比（按模型取最优 trial）"
        extra={<Select size="small" style={{ minWidth: 140 }} value={activeMetric} onChange={setChartMetric}
          options={vm.metricKeys.map(k => ({ value: k, label: k }))} />}>
        {chartData.models.length ? <EChart option={barOption} style={{ height: 280 }} /> : <Empty description="该指标暂无数据" />}
      </Card>

      {task?.id && (
        <Card size="small" title="按策略对比（基线 / 网格 / 贝叶斯）" bodyStyle={{ padding: 12 }}>
          <StrategyCompareTab taskId={task.id} onInspect={(rid) => openInspector(rid, 'shap')} />
        </Card>
      )}

      <Modal open={!!deployModal} title="部署此模型" okText="部署" cancelText="取消" confirmLoading={deploying}
        onCancel={() => setDeployModal(null)}
        onOk={async () => { await deploy(deployModal.runId, { name: deployModal.name }); setDeployModal(null) }}>
        <Text type="secondary" style={{ fontSize: 12 }}>部署名称</Text>
        <Input value={deployModal?.name || ''} onChange={e => setDeployModal(m => ({ ...m, name: e.target.value }))} style={{ marginTop: 6 }} />
      </Modal>

      <RunInspector open={!!inspectorRunId} runId={inspectorRunId} defaultTab={inspectorTab}
        onClose={() => setInspectorRunId(null)} />
    </Space>
  )
}
