import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Alert, Button, Card, Col, Descriptions, Row, Skeleton,
  Space, Statistic, Tabs, Tag, Typography, message,
} from 'antd'
import {
  ArrowLeftOutlined, BarChartOutlined, DownloadOutlined, FileTextOutlined,
  LineChartOutlined, ReloadOutlined, SafetyCertificateOutlined,
} from '@ant-design/icons'

import ResultLogPanel from './ResultLogPanel'
import ModelExplainPanel from './ModelExplainPanel'
import TrainingViz from '../workbench/TrainingViz'
import BacktestPanel from './BacktestPanel'
import ModelDownloadPanel from './ModelDownloadPanel'
import { getResultViewEntries } from './resultViewRegistry'
import { dlApi, modelApi } from '../../services/api'
import { formatDateTime, metricLabels } from '../../utils/formatters'

const { Text, Title } = Typography

const METRIC_PRIORITY = {
  ml: {
    classification: ['accuracy', 'f1', 'precision', 'recall', 'roc_auc'],
    regression: ['r2', 'rmse', 'mae', 'mse'],
  },
  dl: {
    classification: ['val_acc', 'val_f1_macro', 'val_precision', 'val_recall', 'val_auc_roc', 'best_val_loss'],
    regression: ['val_r2', 'val_rmse', 'val_mae', 'val_mape', 'best_val_loss'],
  },
}

function inferTaskType(raw, family) {
  if (raw?.task_type === 'classification' || raw?.task_type === 'regression') return raw.task_type
  const modelType = String(raw?.model_type || '').toLowerCase()
  const metrics = raw?.result_metrics ?? {}
  if (family === 'dl' && ['classification', 'regression'].includes(raw?.task_type)) return raw.task_type
  if (modelType.includes('regress') || ['ridge', 'lasso', 'elasticnet', 'svr'].includes(modelType)) return 'regression'
  if (['r2', 'rmse', 'mae', 'val_r2', 'val_rmse', 'val_mae'].some(key => typeof metrics[key] === 'number')) return 'regression'
  return 'classification'
}

function normalizeResult(raw, family, taskId) {
  const metrics = raw?.result_metrics ?? {}
  const taskType = inferTaskType(raw, family)
  return {
    id: taskId,
    name: raw?.name || raw?.task_name || `${raw?.model_type || 'model'}_${taskId.slice(0, 8)}`,
    modelType: raw?.model_type || '-',
    taskType,
    status: String(raw?.status || 'SUCCESS').toUpperCase(),
    datasetName: raw?.dataset?.name || raw?.dataset_name || raw?.dataset_id || '-',
    targetColumn: raw?.target_column || '-',
    finishedAt: raw?.finished_at || raw?.updated_at || null,
    metrics,
  }
}

function metricValue(key, value) {
  if (typeof value !== 'number') return { value: '-', precision: undefined }
  if (/(^|_)(acc|accuracy|precision|recall|f1|auc)/i.test(key) && Math.abs(value) <= 1) {
    return { value: value * 100, precision: 2, suffix: '%' }
  }
  return { value, precision: 4 }
}

export default function UnifiedResultDetail({ family, taskId }) {
  const navigate = useNavigate()
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState('logs')

  const load = useCallback(async () => {
    setLoading(true)
    try {
      let raw
      if (family === 'dl') {
        const [status, detail] = await Promise.all([
          dlApi.getStatus(taskId),
          modelApi.getModelDetail(taskId),
        ])
        raw = { ...detail, ...status, dataset: detail?.dataset ?? status?.dataset }
      } else {
        raw = await modelApi.getModelDetail(taskId)
      }
      setResult(normalizeResult(raw, family, taskId))
    } catch (err) {
      message.error(err?.response?.data?.detail || '加载模型结果失败')
      setResult(null)
    } finally {
      setLoading(false)
    }
  }, [family, taskId])

  useEffect(() => {
    setActiveTab('logs')
    void load()
  }, [load])

  const metricEntries = useMemo(() => {
    if (!result) return []
    const preferred = METRIC_PRIORITY[family]?.[result.taskType] ?? []
    return preferred
      .filter(key => typeof result.metrics?.[key] === 'number')
      .slice(0, 5)
      .map(key => [key, result.metrics[key]])
  }, [family, result])

  const tabEntries = result ? getResultViewEntries({
    family,
    taskType: result.taskType,
    status: result.status,
  }) : []

  const renderers = result ? {
    logs: (
      <ResultLogPanel family={family} taskId={taskId} status={result.status} />
    ),
    trainingViz: (
      <TrainingViz
        trainingTaskId={taskId}
        modelType={result.modelType}
        taskStatus={result.status}
        family={family}
        taskType={result.taskType}
        history={result.metrics?.history}
        metrics={result.metrics}
      />
    ),
    backtest: (
      <BacktestPanel
        family={family}
        taskId={taskId}
        taskType={result.taskType}
        metrics={result.metrics}
      />
    ),
    download: (
      <ModelDownloadPanel family={family} taskId={taskId} result={result} />
    ),
    explain: (
      <ModelExplainPanel taskId={taskId} modelType={result.modelType} />
    ),
  } : {}

  const tabIcons = {
    logs: <FileTextOutlined />,
    visualization: <BarChartOutlined />,
    backtest: <LineChartOutlined />,
    download: <DownloadOutlined />,
    explain: <SafetyCertificateOutlined />,
  }

  if (loading && !result) {
    return <Card><Skeleton active paragraph={{ rows: 8 }} /></Card>
  }

  if (!result) {
    return (
      <Alert
        type="error"
        showIcon
        message="模型结果不可用"
        action={<Button onClick={() => navigate('/models')}>返回模型管理</Button>}
      />
    )
  }

  return (
    <div>
      <Space style={{ width: '100%', justifyContent: 'space-between', marginBottom: 20 }} wrap>
        <Space>
          <Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/models')}>
            返回模型管理
          </Button>
          <div>
            <Space size={10} wrap>
              <Title level={2} style={{ margin: 0 }}>模型训练结果</Title>
              <Tag color={family === 'dl' ? 'purple' : 'blue'}>
                {family === 'dl' ? '深度学习' : '机器学习'}
              </Tag>
            </Space>
            <Text type="secondary">{result.name}</Text>
          </div>
        </Space>
        <Button icon={<ReloadOutlined />} loading={loading} onClick={() => void load()}>
          刷新
        </Button>
      </Space>

      {result.status === 'FAILED' && (
        <Alert type="error" showIcon message="训练失败" style={{ marginBottom: 16 }} />
      )}

      <Card style={{ marginBottom: 16 }} styles={{ body: { padding: 18 } }}>
        <Descriptions column={{ xs: 1, md: 2, xl: 4 }} size="small">
          <Descriptions.Item label="模型">{result.modelType}</Descriptions.Item>
          <Descriptions.Item label="任务类型">
            {result.taskType === 'regression' ? '回归' : '分类'}
          </Descriptions.Item>
          <Descriptions.Item label="数据集">{result.datasetName}</Descriptions.Item>
          <Descriptions.Item label="目标列">{result.targetColumn}</Descriptions.Item>
          <Descriptions.Item label="完成时间">{formatDateTime(result.finishedAt)}</Descriptions.Item>
          <Descriptions.Item label="任务 ID" span={3}><Text code>{taskId}</Text></Descriptions.Item>
        </Descriptions>
      </Card>

      {metricEntries.length > 0 && (
        <Row gutter={[12, 12]} style={{ marginBottom: 16 }}>
          {metricEntries.map(([key, value]) => {
            const display = metricValue(key, value)
            return (
              <Col xs={12} md={8} xl={Math.max(4, Math.floor(24 / metricEntries.length))} key={key}>
                <Card size="small">
                  <Statistic title={metricLabels[key] ?? key} {...display} />
                </Card>
              </Col>
            )
          })}
        </Row>
      )}

      <Card styles={{ body: { padding: '0 20px 20px' } }}>
        <Tabs
          activeKey={activeTab}
          onChange={setActiveTab}
          items={tabEntries.map(entry => ({
            key: entry.key,
            label: <Space size={6}>{tabIcons[entry.key]}{entry.label}</Space>,
            children: renderers[entry.renderer],
          }))}
        />
      </Card>
    </div>
  )
}
