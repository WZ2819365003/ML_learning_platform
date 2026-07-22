/**
 * ModelDeploy — 模型部署管理
 * 路由: /deploy
 * 功能: ML / DL / TimesFM 三种部署统一管理
 *       - 新增部署弹窗（Modal）
 *       - 详情 + 在线测试 抽屉（Drawer）
 *       - TS 测试 payload 从数据集动态组装
 *       - Chronos 模型状态卡 + 下载/预热按钮
 */
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Alert, Badge, Button, Card, Col, Descriptions, Drawer, Empty,
  Form, Input, Modal, Popconfirm, Row, Select, Space, Spin,
  Statistic, Table, Tabs, Tag, Typography, message,
} from 'antd'
import {
  CloudDownloadOutlined, CloudServerOutlined, CopyOutlined,
  DeleteOutlined, EyeOutlined, PauseCircleOutlined, PlayCircleOutlined,
  PlusOutlined, ReloadOutlined, ThunderboltOutlined,
} from '@ant-design/icons'
import api, { dataApi, deployApi, dlApi, modelApi, trainingApi, tsApi } from '../services/api'
import BatchPredictPanel from '../components/workbench/BatchPredictPanel'
import { formatDateTime } from '../utils/formatters'

const { Text, Title } = Typography

// ── helpers ──────────────────────────────────────────────────────────────────
const statusMeta = (s) => ({
  badge:  s === 'active' ? 'success' : 'default',
  text:   s === 'active' ? '运行中' : '已暂停',
  color:  s === 'active' ? '#52c41a' : '#8c8c8c',
})

// ── 可复制地址框 ───────────────────────────────────────────────────────────────
function CopyField({ label, value }) {
  if (!value) return <Text type="secondary">无可用地址</Text>
  return (
    <div className="ts-copy-field">
      <Text type="secondary" style={{ display: 'block', marginBottom: 6 }}>{label}</Text>
      <Space style={{ width: '100%' }} align="start">
        <Text code style={{ wordBreak: 'break-all', flex: 1 }}>{value}</Text>
        <Button
          size="small" icon={<CopyOutlined />}
          onClick={() => { navigator.clipboard.writeText(value); message.success('已复制') }}
        />
      </Space>
    </div>
  )
}

// ── 统计 + 表格面板（三种部署共用）──────────────────────────────────────────────
function DeployPanel({ stats, loading, data, columns, onRefresh, onNew }) {
  return (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      <Row gutter={[12, 12]}>
        {[
          { label: '部署总数', val: stats.total,  color: '#1890ff' },
          { label: '运行中',   val: stats.active, color: '#52c41a' },
          { label: '累计调用', val: stats.calls,  color: '#722ed1' },
        ].map(({ label, val, color }) => (
          <Col xs={8} key={label}>
            <Card size="small" style={{ textAlign: 'center', borderTop: `3px solid ${color}` }}>
              <Statistic
                title={<Text type="secondary" style={{ fontSize: 12 }}>{label}</Text>}
                value={val}
                valueStyle={{ color, fontSize: 20, fontWeight: 700 }}
              />
            </Card>
          </Col>
        ))}
      </Row>
      <Card
        className="ts-panel-card"
        extra={
          <Space size={8}>
            <Button size="small" icon={<ReloadOutlined />} onClick={onRefresh}>刷新</Button>
            <Button size="small" type="primary" icon={<PlusOutlined />} onClick={onNew}>新增部署</Button>
          </Space>
        }
      >
        <Table
          rowKey={(r) => r.deployment_id ?? r.id}
          columns={columns}
          dataSource={data}
          loading={loading}
          size="middle"
          locale={{ emptyText: <Empty description="暂无部署记录" /> }}
          scroll={{ x: 900 }}
          pagination={{ pageSize: 8, showSizeChanger: false, showTotal: (t) => `共 ${t} 条` }}
        />
      </Card>
    </Space>
  )
}

// ── Chronos 模型状态卡 ─────────────────────────────────────────────────────────
function ChronosStatusCard({ status, statusLoading, preloading, modelName, onModelChange, onRefresh, onPreload }) {
  return (
    <Card
      className="ts-panel-card"
      title="Chronos 模型状态"
      extra={
        <Space wrap>
          <Select
            value={modelName}
            style={{ width: 200 }}
            onChange={onModelChange}
            options={[
              { value: 'amazon/chronos-t5-tiny',  label: 'T5-Tiny（最快）' },
              { value: 'amazon/chronos-t5-small', label: 'T5-Small（默认）' },
              { value: 'amazon/chronos-t5-base',  label: 'T5-Base' },
              { value: 'amazon/chronos-t5-large', label: 'T5-Large（最准）' },
            ]}
          />
          <Button icon={<ReloadOutlined />} loading={statusLoading} onClick={onRefresh}>刷新状态</Button>
          <Button
            type="primary" icon={<CloudDownloadOutlined />}
            loading={preloading} onClick={onPreload}
          >
            下载 / 预热
          </Button>
        </Space>
      }
    >
      {statusLoading && !status ? (
        <Spin tip="加载中…" />
      ) : (
        <Space direction="vertical" style={{ width: '100%' }} size={16}>
          <Row gutter={[12, 12]}>
            {[
              { label: '依赖状态', val: status?.available ? '已安装' : '未安装', color: status?.available ? '#52c41a' : '#ff4d4f' },
              { label: '当前加载', val: status?.loaded    ? '已加载' : '未加载', color: status?.loaded    ? '#1890ff' : '#8c8c8c' },
              { label: '模型名称', val: (status?.model ?? modelName).split('/').pop(), color: '#595959' },
            ].map(({ label, val, color }) => (
              <Col span={8} key={label}>
                <Card size="small" style={{ textAlign: 'center' }}>
                  <Statistic
                    title={<Text type="secondary" style={{ fontSize: 12 }}>{label}</Text>}
                    value={val}
                    valueStyle={{ color, fontSize: 16, fontWeight: 600 }}
                  />
                </Card>
              </Col>
            ))}
          </Row>
          <Alert
            type={status?.available ? 'info' : 'warning'}
            showIcon
            message={status?.available ? '模型下载说明' : '依赖未安装'}
            description={
              status?.available
                ? '选择模型后点击"下载/预热"，首次会从 HuggingFace 下载缓存（tiny ~260MB，small ~460MB，首次约1-5分钟）。后续复用本地缓存。'
                : '请运行：pip install chronos-forecasting torch，安装后重启后端。'
            }
          />
        </Space>
      )}
    </Card>
  )
}

// ── 主组件 ────────────────────────────────────────────────────────────────────
export default function ModelDeploy() {
  // 部署列表
  const [mlList, setMlList] = useState([])
  const [dlList, setDlList] = useState([])
  const [tsList, setTsList] = useState([])
  const [listLoading, setListLoading] = useState({ ml: false, dl: false, ts: false })

  // 抽屉（详情/测试）
  const [drawer, setDrawer] = useState({ open: false, kind: 'ml', record: null, tab: 'overview' })

  // 在线测试状态
  const [tester, setTester] = useState({
    ml: { input: '[\n  {}\n]',  result: null, loading: false, prepared: false },
    dl: { input: '[\n  {}\n]',  result: null, loading: false, prepared: false },
    ts: { input: '{\n  "dataset_id": "",\n  "value_column": "",\n  "time_column": null,\n  "horizon": 24,\n  "frequency": "D"\n}',
          result: null, loading: false, prepared: false },
  })

  // 新建部署弹窗
  const [createModal, setCreateModal] = useState({ kind: null }) // kind: 'ml'|'dl'|'ts'|null
  const [createLoading, setCreateLoading] = useState(false)
  const [mlTasks, setMlTasks] = useState([])
  const [dlTasks, setDlTasks] = useState([])

  // Chronos 状态
  const [tsStatus, setTsStatus]         = useState(null)
  const [tsStatusLoading, setTsStatusLoading] = useState(false)
  const [tsPreloading, setTsPreloading] = useState(false)
  const [tsModelName, setTsModelName]   = useState('amazon/chronos-t5-small')

  const preparingRef = useRef(false)  // debounce payload prep

  // ── fetch helpers ─────────────────────────────────────────────────────────
  const fetchMl = useCallback(async () => {
    setListLoading(s => ({ ...s, ml: true }))
    try { const r = await deployApi.listDeployments(); setMlList(r.deployments ?? []) }
    catch { message.error('获取机器学习部署失败') }
    finally { setListLoading(s => ({ ...s, ml: false })) }
  }, [])

  const fetchDl = useCallback(async () => {
    setListLoading(s => ({ ...s, dl: true }))
    try { const r = await dlApi.listDeployments(); setDlList(r.deployments ?? []) }
    catch { message.error('获取深度学习部署失败') }
    finally { setListLoading(s => ({ ...s, dl: false })) }
  }, [])

  const fetchTs = useCallback(async () => {
    setListLoading(s => ({ ...s, ts: true }))
    try { const r = await tsApi.listDeployments({ page: 1, page_size: 100 }); setTsList(r.items ?? []) }
    catch { message.error('获取 TimesFM 部署失败') }
    finally { setListLoading(s => ({ ...s, ts: false })) }
  }, [])

  const fetchTsStatus = useCallback(async () => {
    setTsStatusLoading(true)
    try {
      const r = await tsApi.modelStatus()
      setTsStatus(r)
      if (r?.model) setTsModelName(r.model)
    } catch (err) {
      // 路由未注册时不弹全局错误，只置空
      console.warn('ts model status failed:', err?.message)
      setTsStatus(null)
    } finally { setTsStatusLoading(false) }
  }, [])

  useEffect(() => {
    void fetchMl()
    void fetchDl()
    void fetchTs()
    void fetchTsStatus()
  }, [fetchMl, fetchDl, fetchTs, fetchTsStatus])

  // ── 准备测试 payload（仅在打开 testing tab 且未准备过时执行）──────────────────
  const preparePayload = useCallback(async (kind, record) => {
    if (preparingRef.current) return
    preparingRef.current = true
    try {
      if (kind === 'ml' && record?.task_id) {
        const detail  = await modelApi.getModelDetail(record.task_id)
        const preview = await dataApi.previewDataset(detail.dataset?.id ?? detail.dataset_id)
        if (preview?.rows?.[0] && detail.target_column) {
          const row = { ...preview.rows[0] }
          delete row[detail.target_column]
          setTester(s => ({ ...s, ml: { ...s.ml, input: JSON.stringify([row], null, 2), prepared: true } }))
        }
      }
      if (kind === 'dl' && record?.dl_task_id) {
        const task    = await dlApi.getStatus(record.dl_task_id)
        const preview = await dataApi.previewDataset(task.dataset_id)
        if (preview?.rows?.[0] && task.target_column) {
          const row = { ...preview.rows[0] }
          delete row[task.target_column]
          setTester(s => ({ ...s, dl: { ...s.dl, input: JSON.stringify([row], null, 2), prepared: true } }))
        }
      }
      if (kind === 'ts') {
        // 动态选择第一份真正像时序的数据集组装 payload
        // 时序预测需要：
        //   time_column  — datetime dtype 或名字含 time/date/timestamp/ds
        //   value_column — 数值型（int/float）且 unique_rate < 0.9（过滤 UDI 这种 ID）
        //   frequency    — 默认 'D'，让用户自己按需改成 H/T/M/Q/Y
        const dsRes = await dataApi.listDatasets({ page: 1, page_size: 20 })
        const datasets = dsRes.items ?? dsRes.datasets ?? []
        if (datasets.length === 0) return
        let selectedDs = null
        let valueCol = null
        let timeCol  = null

        const inferColumns = (preview, ds) => {
          const colsInfo = preview?.columns_info ?? {}
          const stats    = preview?.statistics ?? {}
          const rowCount = preview?.row_count || ds.row_count || 0
          const cols     = Object.keys(colsInfo)

          const isDatetime = (name) => {
            const dt = String(colsInfo[name]?.dtype ?? '').toLowerCase()
            if (dt.startsWith('datetime') || dt.includes('date')) return true
            return ['time','date','datetime','timestamp','ds'].includes(name.toLowerCase())
          }
          const isNumericNonId = (name) => {
            const dt = String(colsInfo[name]?.dtype ?? '').toLowerCase()
            const numeric = dt.startsWith('int') || dt.startsWith('float') || dt.startsWith('uint')
            if (!numeric) return false
            const unique = Number(stats[name]?.unique_count ?? 0)
            const rate   = rowCount > 0 ? unique / rowCount : 0
            return rate < 0.9  // ID-like columns (UDI etc.) are filtered out
          }

          timeCol  = cols.find(isDatetime) ?? null
          valueCol = cols.find(c => c !== timeCol && isNumericNonId(c)) ?? null
          return { timeCol, valueCol }
        }

        for (const ds of datasets) {
          try {
            const preview = await dataApi.previewDataset(ds.id)
            const inferred = inferColumns(preview, ds)
            if (inferred.timeCol && inferred.valueCol) {
              selectedDs = ds
              timeCol = inferred.timeCol
              valueCol = inferred.valueCol
              break
            }
          } catch { /* try next dataset */ }
        }
        const payload = {
          dataset_id: selectedDs?.id ?? '',
          value_column: valueCol ?? '',
          time_column: timeCol ?? '',
          horizon: 24,
          frequency: 'D',
        }
        setTester(s => ({ ...s, ts: { ...s.ts, input: JSON.stringify(payload, null, 2), prepared: true } }))
      }
    } catch { /* silent */ }
    finally { preparingRef.current = false }
  }, [])

  // 打开抽屉时重置 prepared 标志，切换到 testing tab 时触发 prepare
  const openDrawer = useCallback((kind, record, tab = 'overview') => {
    setTester(s => ({
      ...s,
      [kind]: { ...s[kind], result: null, prepared: false },
    }))
    setDrawer({ open: true, kind, record, tab })
    if (tab === 'testing') {
      void preparePayload(kind, record)
    }
  }, [preparePayload])

  const closeDrawer = () => setDrawer({ open: false, kind: 'ml', record: null, tab: 'overview' })

  // tab 切换时，仅在切到 testing 且未准备时触发 prepare
  const onDrawerTabChange = (tab) => {
    setDrawer(s => ({ ...s, tab }))
    if (tab === 'testing' && !tester[drawer.kind]?.prepared) {
      void preparePayload(drawer.kind, drawer.record)
    }
  }

  // ── Chronos 预热 ───────────────────────────────────────────────────────────
  const handlePreload = async () => {
    setTsPreloading(true)
    try {
      const r = await tsApi.preloadModel(tsModelName)
      message.success(r?.message ?? `已开始后台下载 ${tsModelName.split('/').pop()}，请稍候…`)
      // 3 秒后刷新状态
      setTimeout(() => void fetchTsStatus(), 3000)
    } catch (err) {
      message.error(err?.response?.data?.detail ?? '触发模型下载失败，请确认后端已重启')
    } finally { setTsPreloading(false) }
  }

  // ── 切换部署状态 ──────────────────────────────────────────────────────────
  const handleToggle = async (kind, record) => {
    const next = record.status === 'active' ? 'paused' : 'active'
    try {
      if      (kind === 'ml') await deployApi.updateStatus(record.deployment_id, next)
      else if (kind === 'dl') await dlApi.toggleDeployment(record.id, next)
      else                    await tsApi.updateDeploymentStatus(record.deployment_id, next)
      message.success(next === 'active' ? '部署已启用' : '部署已暂停')
      if (kind === 'ml') void fetchMl()
      else if (kind === 'dl') void fetchDl()
      else void fetchTs()
    } catch (err) { message.error(err?.response?.data?.detail ?? '切换状态失败') }
  }

  // ── 删除部署 ───────────────────────────────────────────────────────────────
  const handleDelete = async (kind, record) => {
    try {
      if      (kind === 'ml') await deployApi.deleteDeployment(record.deployment_id)
      else if (kind === 'dl') await dlApi.deleteDeployment(record.id)
      else                    await tsApi.deleteDeployment(record.deployment_id)
      message.success('部署已删除')
      const rid = record.deployment_id ?? record.id
      const vid = drawer.record?.deployment_id ?? drawer.record?.id
      if (rid === vid) closeDrawer()
      if (kind === 'ml') void fetchMl()
      else if (kind === 'dl') void fetchDl()
      else void fetchTs()
    } catch (err) { message.error(err?.response?.data?.detail ?? '删除失败') }
  }

  // ── 在线测试 ───────────────────────────────────────────────────────────────
  const handleTest = async () => {
    if (!drawer.record) return
    const kind = drawer.kind
    setTester(s => ({ ...s, [kind]: { ...s[kind], loading: true, result: null } }))
    try {
      let result
      if (kind === 'ml') {
        const rows = JSON.parse(tester[kind].input)
        result = await deployApi.predict(drawer.record.deployment_id, {
          rows: Array.isArray(rows) ? rows : [rows], include_probabilities: true,
        })
      } else if (kind === 'dl') {
        const rows = JSON.parse(tester[kind].input)
        result = await dlApi.predictDeployment(drawer.record.id, {
          rows: Array.isArray(rows) ? rows : [rows],
        })
      } else {
        const payload = JSON.parse(tester[kind].input)
        if (!payload.dataset_id || !payload.value_column || !payload.time_column) {
          throw new Error('时序测试需要 dataset_id、value_column 和 time_column，请先补齐参数')
        }
        result = await tsApi.predictDeployment(drawer.record.deployment_id, payload)
      }
      setTester(s => ({ ...s, [kind]: { ...s[kind], result } }))
    } catch (err) {
      message.error(err?.response?.data?.detail ?? err?.message ?? '在线测试失败')
    } finally {
      setTester(s => ({ ...s, [kind]: { ...s[kind], loading: false } }))
    }
  }

  // ── 新建部署 ───────────────────────────────────────────────────────────────
  const openCreate = async (kind) => {
    if (kind === 'ml') {
      const r = await trainingApi.listTasks({ page: 1, page_size: 100 }).catch(() => ({ items: [] }))
      setMlTasks((r.items ?? r.tasks ?? []).filter(t => t.status === 'SUCCESS' || t.status === 'success'))
    }
    if (kind === 'dl') {
      const r = await dlApi.listTasks({ page: 1, page_size: 100 }).catch(() => ({ items: [] }))
      setDlTasks((r.items ?? []).filter(t => t.status === 'SUCCESS' || t.status === 'success'))
    }
    setCreateModal({ kind })
  }

  const closeCreate = () => setCreateModal({ kind: null })

  const handleCreate = async (values) => {
    const kind = createModal.kind
    setCreateLoading(true)
    try {
      if (kind === 'ml') {
        await deployApi.createDeployment(values.task_id, { name: values.name, description: values.description ?? '' })
      } else if (kind === 'dl') {
        await dlApi.createDeployment(values.task_id, { name: values.name, description: values.description ?? '' })
      } else {
        await tsApi.createDeployment({ name: values.name, backend_label: values.backend_label, description: values.description ?? '' })
      }
      message.success('部署已创建')
      closeCreate()
      if (kind === 'ml') void fetchMl()
      else if (kind === 'dl') void fetchDl()
      else { void fetchTs(); void fetchTsStatus() }
    } catch (err) { message.error(err?.response?.data?.detail ?? '创建部署失败') }
    finally { setCreateLoading(false) }
  }

  // ── 辅助：预测 URL ────────────────────────────────────────────────────────
  const getPredictUrl = (kind, record) => {
    if (!record) return ''
    if (kind === 'ml') return record.endpoints?.predict ?? ''
    if (kind === 'dl') return `${api.defaults.baseURL}/dl/deployments/${record.id}/predict`
    return record.predict_url ?? `${api.defaults.baseURL}/ts/deployments/${record.deployment_id}/predict`
  }

  // ── 详情 Descriptions ─────────────────────────────────────────────────────
  const getDetailItems = (kind, record) => {
    if (!record) return []
    const m = statusMeta(record.status)
    const common = [
      { key: 'status',   label: '状态',    children: <Badge status={m.badge} text={<Text style={{ color: m.color }}>{m.text}</Text>} /> },
      { key: 'created',  label: '创建时间', children: formatDateTime(record.created_at) },
      { key: 'calls',    label: '调用次数', children: record.request_count ?? 0 },
      { key: 'desc',     label: '说明',     children: record.description || '—' },
    ]
    if (kind === 'ml') return [...common, { key: 'tid', label: '训练任务 ID', children: <Text copyable style={{ fontSize: 12 }}>{record.task_id ?? '—'}</Text> }]
    if (kind === 'dl') return [...common, { key: 'tid', label: '训练任务 ID', children: <Text copyable style={{ fontSize: 12 }}>{record.dl_task_id ?? '—'}</Text> }]
    return [...common, { key: 'be', label: '基础模型', children: <Tag color="blue">{record.backend_label?.split('/').pop() ?? '—'}</Tag> }]
  }

  // ── 表格列定义 ────────────────────────────────────────────────────────────
  const buildColumns = (kind) => [
    {
      title: '部署名称',
      dataIndex: 'name',
      render: (v, r) => (
        <Space direction="vertical" size={2}>
          <Button type="link" style={{ padding: 0, fontWeight: 600 }} onClick={() => openDrawer(kind, r, 'overview')}>{v}</Button>
          {kind === 'ts' && r.backend_label && (
            <Tag color="blue" style={{ fontSize: 10 }}>{r.backend_label.split('/').pop()}</Tag>
          )}
        </Space>
      ),
    },
    {
      title: '状态', dataIndex: 'status', width: 100,
      render: v => { const m = statusMeta(v); return <Badge status={m.badge} text={<Text style={{ color: m.color }}>{m.text}</Text>} /> },
    },
    { title: '调用次数', dataIndex: 'request_count', width: 95, render: v => v ?? 0 },
    { title: '创建时间', dataIndex: 'created_at', width: 145, render: v => <Text style={{ fontSize: 12 }}>{formatDateTime(v)}</Text> },
    {
      title: '操作', key: 'act', width: 210,
      render: (_, r) => (
        <Space size={4} onClick={e => e.stopPropagation()}>
          <Button size="small" icon={<EyeOutlined />} onClick={() => openDrawer(kind, r, 'overview')}>详情</Button>
          <Button size="small" icon={<ThunderboltOutlined />} onClick={() => openDrawer(kind, r, 'testing')}>测试</Button>
          <Button
            size="small"
            icon={r.status === 'active' ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
            onClick={() => void handleToggle(kind, r)}
          />
          <Popconfirm title="确认删除此部署？" okButtonProps={{ danger: true }} onConfirm={() => void handleDelete(kind, r)}>
            <Button size="small" danger icon={<DeleteOutlined />} />
          </Popconfirm>
        </Space>
      ),
    },
  ]

  // ── 统计 ──────────────────────────────────────────────────────────────────
  const calcStats = (list) => ({
    total:  list.length,
    active: list.filter(x => x.status === 'active').length,
    calls:  list.reduce((s, x) => s + (x.request_count ?? 0), 0),
  })
  const mlStats = useMemo(() => calcStats(mlList), [mlList])
  const dlStats = useMemo(() => calcStats(dlList), [dlList])
  const tsStats = useMemo(() => calcStats(tsList), [tsList])

  // ── 新建弹窗：内容 ─────────────────────────────────────────────────────────
  const CreateModalContent = () => {
    const [form] = Form.useForm()
    const kind = createModal.kind
    if (!kind) return null

    const taskOptions = kind === 'ml'
      ? mlTasks.map(t => ({ value: t.id ?? t.task_id, label: `${t.task_name ?? t.name ?? t.id?.slice(0,8)} · ${t.model_type ?? ''}` }))
      : dlTasks.map(t => ({ value: t.id ?? t.task_id, label: `${t.task_name ?? t.name ?? t.id?.slice(0,8)} · ${t.model_type ?? ''}` }))

    return (
      <Modal
        title={kind === 'ml' ? '新增机器学习部署' : kind === 'dl' ? '新增深度学习部署' : '新增 TimesFM 部署'}
        open={!!kind}
        onCancel={closeCreate}
        confirmLoading={createLoading}
        onOk={() => form.validateFields().then(v => handleCreate(v))}
        okText="创建部署"
        destroyOnClose
      >
        <Form form={form} layout="vertical" style={{ marginTop: 8 }}>
          {(kind === 'ml' || kind === 'dl') && (
            <Form.Item
              label="训练任务" name="task_id"
              rules={[{ required: true, message: '请选择训练完成的任务' }]}
              extra={taskOptions.length === 0 ? <Text type="warning">暂无已完成的训练任务</Text> : null}
            >
              <Select
                showSearch optionFilterProp="label"
                placeholder={`选择已完成的${kind === 'ml' ? '机器学习' : '深度学习'}任务`}
                options={taskOptions}
              />
            </Form.Item>
          )}
          {kind === 'ts' && (
            <Form.Item label="基础模型" name="backend_label" initialValue="amazon/chronos-t5-small">
              <Select
                options={[
                  { value: 'amazon/chronos-t5-tiny',  label: 'Chronos T5-Tiny（最快）' },
                  { value: 'amazon/chronos-t5-small', label: 'Chronos T5-Small（默认）' },
                  { value: 'amazon/chronos-t5-base',  label: 'Chronos T5-Base' },
                  { value: 'amazon/chronos-t5-large', label: 'Chronos T5-Large（最准）' },
                ]}
              />
            </Form.Item>
          )}
          <Form.Item label="部署名称" name="name" rules={[{ required: true, message: '请输入部署名称' }]}>
            <Input placeholder={kind === 'ts' ? '例如：Chronos 生产服务 A' : '例如：XGBoost 生产部署 v1'} />
          </Form.Item>
          <Form.Item label="说明" name="description">
            <Input.TextArea rows={3} placeholder="记录部署用途、调用方或版本说明" />
          </Form.Item>
        </Form>
      </Modal>
    )
  }

  // ── render ────────────────────────────────────────────────────────────────
  return (
    <Space direction="vertical" size={20} style={{ width: '100%' }}>
      {/* 页头 */}
      <div className="ts-page-hero">
        <Space direction="vertical" size={4}>
          <Title level={2} style={{ margin: 0 }}>
            <Space><CloudServerOutlined />模型部署</Space>
          </Title>
          <Text type="secondary">统一管理机器学习、深度学习和 Chronos 时序服务的在线部署实例</Text>
        </Space>
      </div>

      {/* Tab 面板 */}
      <Tabs
        defaultActiveKey="ml"
        items={[
          {
            key: 'ml',
            label: '机器学习',
            children: (
              <DeployPanel
                stats={mlStats} loading={listLoading.ml} data={mlList}
                columns={buildColumns('ml')}
                onRefresh={() => void fetchMl()}
                onNew={() => void openCreate('ml')}
              />
            ),
          },
          {
            key: 'dl',
            label: '深度学习',
            children: (
              <DeployPanel
                stats={dlStats} loading={listLoading.dl} data={dlList}
                columns={buildColumns('dl')}
                onRefresh={() => void fetchDl()}
                onNew={() => void openCreate('dl')}
              />
            ),
          },
          {
            key: 'timesfm',
            label: 'TimesFM / Chronos',
            children: (
              <Space direction="vertical" size={16} style={{ width: '100%' }}>
                <ChronosStatusCard
                  status={tsStatus}
                  statusLoading={tsStatusLoading}
                  preloading={tsPreloading}
                  modelName={tsModelName}
                  onModelChange={setTsModelName}
                  onRefresh={() => void fetchTsStatus()}
                  onPreload={() => void handlePreload()}
                />
                <DeployPanel
                  stats={tsStats} loading={listLoading.ts} data={tsList}
                  columns={buildColumns('ts')}
                  onRefresh={() => void fetchTs()}
                  onNew={() => void openCreate('ts')}
                />
              </Space>
            ),
          },
        ]}
      />

      {/* 详情 / 测试 抽屉 */}
      <Drawer
        title={drawer.record ? <Space><CloudServerOutlined />{drawer.record.name}</Space> : '部署详情'}
        open={drawer.open} onClose={closeDrawer} width={720} destroyOnClose
      >
        {!drawer.record ? (
          <Empty description="请选择一个部署实例" />
        ) : (
          <Tabs
            activeKey={drawer.tab}
            destroyOnHidden
            onChange={onDrawerTabChange}
            items={[
              {
                key: 'overview',
                label: '概览',
                children: (
                  <Descriptions bordered column={1} size="small"
                    items={getDetailItems(drawer.kind, drawer.record)}
                  />
                ),
              },
              {
                key: 'endpoints',
                label: '服务地址',
                children: (
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <CopyField label="预测接口 URL" value={getPredictUrl(drawer.kind, drawer.record)} />
                  </Space>
                ),
              },
              {
                key: 'testing',
                label: '在线测试',
                children: (
                  <Space direction="vertical" size={16} style={{ width: '100%' }}>
                    <Alert
                      type="info" showIcon message="在线测试说明"
                      description={
                        drawer.kind === 'ts'
                          ? 'TimesFM 请求需要 dataset_id、value_column、time_column、horizon、frequency。系统只会自动填入同时具备时间列和数值目标列的数据集；若未推断到，请手工补齐，避免把 UDI 等 ID 列当作预测值。'
                          : 'ML / DL 请求已根据训练数据集自动预填第一行样本（已删除目标列），可直接点发送。'
                      }
                    />
                    <Input.TextArea
                      rows={10}
                      value={tester[drawer.kind]?.input ?? ''}
                      onChange={e => setTester(s => ({ ...s, [drawer.kind]: { ...s[drawer.kind], input: e.target.value } }))}
                      style={{ fontFamily: 'Consolas, Monaco, monospace', fontSize: 12 }}
                    />
                    <div className="ts-drawer-footer">
                      <Button
                        type="primary" icon={<ThunderboltOutlined />}
                        loading={tester[drawer.kind]?.loading}
                        disabled={drawer.record.status !== 'active'}
                        onClick={() => void handleTest()}
                      >
                        发送测试请求
                      </Button>
                    </div>
                    {tester[drawer.kind]?.result && (
                      <pre className="ts-json-block">
                        {JSON.stringify(tester[drawer.kind].result, null, 2)}
                      </pre>
                    )}
                  </Space>
                ),
              },
              // ML deployments only: batch prediction is backed by the classical
              // inference path (deploy_service + InferenceJob). DL/TS deployments
              // have their own predict routes and no batch job table.
              ...(drawer.kind === 'ml' && drawer.record ? [{
                key: 'batch',
                label: '批量预测',
                // /deploy/list returns `deployment_id`, not `id` — passing record.id
                // silently sends `undefined` and the API answers 「部署不存在」.
                children: <BatchPredictPanel deploymentId={drawer.record.deployment_id} />,
              }] : []),
            ]}
          />
        )}
      </Drawer>

      {/* 新建部署弹窗 */}
      <CreateModalContent />
    </Space>
  )
}
