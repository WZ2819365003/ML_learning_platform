import React, { useMemo, useState } from 'react'
import {
  Card, Table, Tag, Button, Space, Tooltip, Typography, Empty, Alert, Spin,
  Select, Modal, Input, message,
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
import EChart from '../EChart'
import RunInspector from './RunInspector'
import { useNavigate } from 'react-router-dom'
import StrategyCompareTab from './StrategyCompareTab'

const { Text } = Typography

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
  const [finalizing, setFinalizing] = useState(false)
  const { deploying, error: deployError, deploy, reset: resetDeploy } = useDeployRun(task)

  const navigate = useNavigate()

  // 详情 opens the standalone page and tells it where 返回 should go; the
  // drawer stays for 解释, which is a peek rather than a destination.
  const openDetailPage = (rid) => {
    navigate(`/models/${rid}`, { state: { from: 'workflow', taskId: task?.id } })
  }
  const openInspector = (rid, tab = 'overview') => { setInspectorRunId(rid); setInspectorTab(tab) }
  const bestRun = vm.rows.find(r => r.is_best)
  const finalization = useMemo(
    () => buildFinalizationVM(task, bestRun),
    [task, bestRun],
  )
  const activeMetric = chartMetric && vm.metricKeys.includes(chartMetric) ? chartMetric : vm.objective_metric

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
  const chartData = useMemo(() => {
    const byModel = {}
    vm.rows.filter(r => r.is_success).forEach(r => {
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
        return typeof v === 'number'
          ? <code style={{ color: k === vm.objective_metric ? '#2563eb' : '#334155', fontWeight: k === vm.objective_metric ? 600 : 400 }}>{v.toFixed(4)}</code>
          : <span style={{ color: '#cbd5e1' }}>-</span>
      },
    })),
    { title: '操作', key: 'actions', width: 220,
      render: (_, r) => (
        <Space size={2}>
          <Button size="small" type="link" onClick={() => openDetailPage(r.run_id)}>详情</Button>
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
    <Space direction="vertical" size={12} style={{ width: '100%' }}>
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
