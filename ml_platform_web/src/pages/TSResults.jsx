import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import {
  Alert,
  Badge,
  Button,
  Card,
  Col,
  Descriptions,
  Empty,
  Form,
  Input,
  Modal,
  Popconfirm,
  Row,
  Select,
  Space,
  Spin,
  Statistic,
  Table,
  Tag,
  Typography,
  message,
} from 'antd'
import {
  ArrowLeftOutlined,
  DeleteOutlined,
  EditOutlined,
  LineChartOutlined,
  ReloadOutlined,
} from '@ant-design/icons'
import ForecastChart from '../components/timeseries/ForecastChart'
import { modelApi, tsApi } from '../services/api'
import { formatDateTime } from '../utils/formatters'

const { Title, Text } = Typography
const { TextArea } = Input

const STATUS_META = {
  PENDING: { badge: 'default', label: '待执行', color: '#8c8c8c' },
  RUNNING: { badge: 'processing', label: '运行中', color: '#faad14' },
  SUCCESS: { badge: 'success', label: '已完成', color: '#52c41a' },
  FAILED: { badge: 'error', label: '失败', color: '#ff4d4f' },
}

const FREQ_LABELS = {
  high: '高频',
  medium: '中频',
  low: '低频',
}

function EditMetaModal({ task, open, onClose, onSaved }) {
  const [form] = Form.useForm()
  const [tagOptions, setTagOptions] = useState([])
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    if (!open || !task) return
    form.setFieldsValue({
      tags: task.tags ?? [],
      notes: task.notes ?? '',
    })
    modelApi
      .listTags()
      .then((response) => {
        const tags = response?.tags ?? response ?? []
        setTagOptions(tags.map((item) => ({ value: item.name ?? item, label: item.name ?? item })))
      })
      .catch(() => setTagOptions([]))
  }, [form, open, task])

  async function handleSave() {
    if (!task?.id) return
    const values = form.getFieldsValue()
    setSaving(true)
    try {
      const payload = {
        tags: values.tags ?? [],
        notes: values.notes ?? '',
      }
      const response = await tsApi.updateTaskMeta(task.id, payload)
      onSaved(response)
      message.success('任务标签与备注已保存')
      onClose()
    } catch (error) {
      message.error(error?.response?.data?.detail ?? '保存任务信息失败')
    } finally {
      setSaving(false)
    }
  }

  return (
    <Modal
      title="编辑任务标签与备注"
      open={open}
      onCancel={onClose}
      onOk={() => void handleSave()}
      confirmLoading={saving}
      okText="保存"
      destroyOnClose
    >
      <Form form={form} layout="vertical">
        <Form.Item label="任务标签" name="tags">
          <Select
            mode="tags"
            placeholder="输入标签后回车"
            options={tagOptions}
          />
        </Form.Item>
        <Form.Item label="备注" name="notes">
          <TextArea rows={4} maxLength={500} showCount placeholder="填写任务说明、验收备注或业务说明" />
        </Form.Item>
      </Form>
    </Modal>
  )
}

export default function TSResults() {
  const { taskId } = useParams()
  const navigate = useNavigate()

  const [task, setTask] = useState(null)
  const [loading, setLoading] = useState(true)
  const [editOpen, setEditOpen] = useState(false)
  const [resultView, setResultView] = useState('chart')
  const pollTimer = useRef(null)

  const fetchTask = useCallback(async () => {
    if (!taskId) {
      setLoading(false)
      return null
    }

    try {
      const response = await tsApi.getTask(taskId)
      setTask(response)
      return response
    } catch (error) {
      message.error(error?.response?.data?.detail ?? '获取时序任务详情失败')
      return null
    } finally {
      setLoading(false)
    }
  }, [taskId])

  useEffect(() => {
    void fetchTask()
    return () => window.clearInterval(pollTimer.current)
  }, [fetchTask])

  useEffect(() => {
    window.clearInterval(pollTimer.current)
    if (!task || !['PENDING', 'RUNNING'].includes(task.status)) {
      return undefined
    }

    pollTimer.current = window.setInterval(() => {
      void fetchTask()
    }, 3000)

    return () => window.clearInterval(pollTimer.current)
  }, [fetchTask, task])

  async function handleDelete() {
    if (!taskId) return
    try {
      await tsApi.deleteTask(taskId)
      message.success('任务已删除')
      navigate('/ts/tasks')
    } catch (error) {
      message.error(error?.response?.data?.detail ?? '删除任务失败')
    }
  }

  const meta = STATUS_META[task?.status] ?? { badge: 'default', label: task?.status ?? '未知', color: '#8c8c8c' }
  const result = task?.result ?? null
  const predictions = result?.predictions ?? result?.point_forecast ?? []
  const q10 = result?.quantile_10 ?? result?.q10 ?? []
  const q90 = result?.quantile_90 ?? result?.q90 ?? []
  const historical = result?.historical_values ?? result?.historical ?? []
  const hasPredictions = predictions.length > 0

  const resultRows = useMemo(
    () =>
      predictions.map((value, index) => ({
        key: index + 1,
        step: index + 1,
        value,
        q10: q10[index],
        q90: q90[index],
      })),
    [predictions, q10, q90],
  )

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', paddingTop: 120 }}>
        <Spin size="large" tip="正在加载任务详情" />
      </div>
    )
  }

  if (!task) {
    return (
      <Empty description="任务不存在或已被删除">
        <Button type="primary" onClick={() => navigate('/ts/tasks')}>
          返回任务列表
        </Button>
      </Empty>
    )
  }

  return (
    <Space direction="vertical" size={20} style={{ width: '100%' }}>
      <Row align="middle" justify="space-between" wrap={false}>
        <Col>
          <Space align="center" size={12}>
            <Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/ts/tasks')}>
              返回列表
            </Button>
            <div>
              <Title level={3} style={{ margin: 0 }}>{task.name || task.dataset_name || '时序任务'}</Title>
              <Space size={8} style={{ marginTop: 4 }}>
                <Badge status={meta.badge} text={<Text style={{ color: meta.color }}>{meta.label}</Text>} />
                {(task.tags ?? []).map((tag) => (
                  <Tag key={tag} color="gold">{tag}</Tag>
                ))}
              </Space>
            </div>
          </Space>
        </Col>
        <Col>
          <Space>
            <Button icon={<ReloadOutlined />} onClick={() => void fetchTask()}>
              刷新
            </Button>
            <Button icon={<EditOutlined />} onClick={() => setEditOpen(true)}>
              编辑
            </Button>
            <Popconfirm
              title="确认删除这个任务？"
              onConfirm={() => void handleDelete()}
              okButtonProps={{ danger: true }}
              disabled={task.status === 'RUNNING'}
            >
              <Button danger icon={<DeleteOutlined />} disabled={task.status === 'RUNNING'}>
                删除
              </Button>
            </Popconfirm>
          </Space>
        </Col>
      </Row>

      <Row gutter={[12, 12]}>
        {[
          { label: '状态', value: meta.label, color: meta.color },
          { label: '目标列', value: task.value_column ?? '—', color: '#1890ff' },
          { label: '预测步长', value: `${task.horizon ?? '—'} 步`, color: '#722ed1' },
          { label: '频率档位', value: FREQ_LABELS[task.frequency] ?? task.frequency ?? '—', color: '#13c2c2' },
        ].map((item) => (
          <Col xs={12} sm={6} key={item.label}>
            <Card size="small" style={{ textAlign: 'center', borderTop: `3px solid ${item.color}` }}>
              <Statistic
                title={<Text type="secondary" style={{ fontSize: 12 }}>{item.label}</Text>}
                value={item.value}
                valueStyle={{ color: item.color, fontSize: 16, fontWeight: 700 }}
              />
            </Card>
          </Col>
        ))}
      </Row>

      <Card size="small" title="任务摘要">
        <Descriptions bordered column={{ xs: 1, sm: 2, md: 3 }} size="small">
          <Descriptions.Item label="任务 ID">
            <Text copyable style={{ fontSize: 12, fontFamily: 'monospace' }}>{task.id}</Text>
          </Descriptions.Item>
          <Descriptions.Item label="数据集">{task.dataset_name ?? task.dataset_id ?? '—'}</Descriptions.Item>
          <Descriptions.Item label="时间列">{task.time_column ?? '按行序号预测'}</Descriptions.Item>
          <Descriptions.Item label="部署实例">{task.deployment_name ?? '默认部署'}</Descriptions.Item>
          <Descriptions.Item label="预测后端">{task.backend_label ?? task.model_name ?? '—'}</Descriptions.Item>
          <Descriptions.Item label="服务地址">
            {task.predict_url ? <Text code copyable>{task.predict_url}</Text> : '—'}
          </Descriptions.Item>
          <Descriptions.Item label="创建时间">{formatDateTime(task.created_at)}</Descriptions.Item>
          <Descriptions.Item label="开始时间">{formatDateTime(task.started_at)}</Descriptions.Item>
          <Descriptions.Item label="完成时间">{formatDateTime(task.finished_at)}</Descriptions.Item>
          {task.error_message ? (
            <Descriptions.Item label="错误信息" span={3}>
              <Text type="danger">{task.error_message}</Text>
            </Descriptions.Item>
          ) : null}
          {task.notes ? (
            <Descriptions.Item label="备注" span={3}>
              {task.notes}
            </Descriptions.Item>
          ) : null}
        </Descriptions>
      </Card>

      {['PENDING', 'RUNNING'].includes(task.status) ? (
        <Alert
          type="info"
          showIcon
          message="任务仍在后台执行"
          description="这个接口返回 RUNNING + result=null 是正常现象，表示后台线程还没完成。首次加载 Chronos 模型时会先下载并初始化权重，耗时会明显更长。页面每 3 秒会自动刷新一次。"
        />
      ) : null}

      {hasPredictions ? (
        <Card
          title={<Space><LineChartOutlined />预测结果</Space>}
          extra={(
            <Space>
              <Button type={resultView === 'chart' ? 'primary' : 'default'} size="small" onClick={() => setResultView('chart')}>
                图表
              </Button>
              <Button type={resultView === 'table' ? 'primary' : 'default'} size="small" onClick={() => setResultView('table')}>
                表格
              </Button>
            </Space>
          )}
        >
          {resultView === 'chart' ? (
            <ForecastChart result={result} />
          ) : (
            <Table
              rowKey="key"
              dataSource={resultRows}
              pagination={{ pageSize: 12, showSizeChanger: false }}
              columns={[
                { title: '步数', dataIndex: 'step', width: 90 },
                {
                  title: '预测值',
                  dataIndex: 'value',
                  render: (value) => Number(value).toFixed(4),
                },
                {
                  title: 'Q10',
                  dataIndex: 'q10',
                  render: (value) => (value == null ? '—' : Number(value).toFixed(4)),
                },
                {
                  title: 'Q90',
                  dataIndex: 'q90',
                  render: (value) => (value == null ? '—' : Number(value).toFixed(4)),
                },
              ]}
            />
          )}

          <Row gutter={16} style={{ marginTop: 16, paddingTop: 12, borderTop: '1px solid #f0f0f0' }}>
            <Col span={8}>
              <Statistic
                title={<Text type="secondary" style={{ fontSize: 11 }}>历史点数</Text>}
                value={historical.length}
                valueStyle={{ fontSize: 18 }}
              />
            </Col>
            <Col span={8}>
              <Statistic
                title={<Text type="secondary" style={{ fontSize: 11 }}>预测点数</Text>}
                value={predictions.length}
                valueStyle={{ fontSize: 18 }}
              />
            </Col>
            <Col span={8}>
              <Statistic
                title={<Text type="secondary" style={{ fontSize: 11 }}>区间信息</Text>}
                value={q10.length ? 'Q10 ~ Q90' : '—'}
                valueStyle={{ fontSize: 14 }}
              />
            </Col>
          </Row>
        </Card>
      ) : task.status === 'FAILED' ? (
        <Card>
          <Empty description="预测任务执行失败" />
        </Card>
      ) : (
        <Card>
          <Empty description="预测结果尚未生成" />
        </Card>
      )}

      <EditMetaModal
        task={task}
        open={editOpen}
        onClose={() => setEditOpen(false)}
        onSaved={(updated) => setTask(updated)}
      />
    </Space>
  )
}
