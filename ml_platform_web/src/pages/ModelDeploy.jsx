import React, { useEffect, useMemo, useState } from 'react'
import {
  Alert, Badge, Button, Card, Descriptions, Drawer, Empty, Form, Input,
  Modal, Popconfirm, Select, Space, Statistic, Table, Tabs, Tag, Typography, message,
} from 'antd'
import {
  CloudDownloadOutlined, CloudServerOutlined, CopyOutlined, DeleteOutlined,
  EyeOutlined, PauseCircleOutlined, PlayCircleOutlined, PlusOutlined,
  ReloadOutlined, ThunderboltOutlined,
} from '@ant-design/icons'
import api, { dataApi, deployApi, dlApi, modelApi, trainingApi, tsApi } from '../services/api'
import { formatDateTime } from '../utils/formatters'

const { Text, Title } = Typography

const statusMeta = (status) => ({
  badge: status === 'active' ? 'success' : 'default',
  text: status === 'active' ? '运行中' : '已暂停',
  color: status === 'active' ? '#52c41a' : '#8c8c8c',
})

function CopyField({ label, value }) {
  if (!value) return null
  return (
    <div className="ts-copy-field">
      <Text type="secondary" style={{ display: 'block', marginBottom: 6 }}>{label}</Text>
      <Space style={{ width: '100%' }} align="start">
        <Text code style={{ wordBreak: 'break-all', flex: 1 }}>{value}</Text>
        <Button
          size="small"
          icon={<CopyOutlined />}
          onClick={() => { navigator.clipboard.writeText(value); message.success('已复制到剪贴板') }}
        />
      </Space>
    </div>
  )
}

function CreateDeploymentModal({ open, onClose, onSubmit, loading, title, fields }) {
  const [form] = Form.useForm()
  useEffect(() => { if (!open) form.resetFields() }, [form, open])
  return (
    <Modal
      title={title}
      open={open}
      onCancel={onClose}
      onOk={async () => { const values = await form.validateFields(); await onSubmit(values); }}
      okText="创建部署"
      confirmLoading={loading}
      destroyOnClose
    >
      <Form form={form} layout="vertical" style={{ marginTop: 8 }}>
        {fields}
      </Form>
    </Modal>
  )
}

function TimesFmCard({ status, loading, preloading, modelName, setModelName, refresh, preload }) {
  return (
    <Card
      className="ts-panel-card"
      extra={(
        <Space>
          <Select
            value={modelName}
            style={{ width: 220 }}
            onChange={setModelName}
            options={[
              { value: 'amazon/chronos-t5-tiny', label: 'Chronos T5-Tiny' },
              { value: 'amazon/chronos-t5-small', label: 'Chronos T5-Small' },
              { value: 'amazon/chronos-t5-base', label: 'Chronos T5-Base' },
              { value: 'amazon/chronos-t5-large', label: 'Chronos T5-Large' },
            ]}
          />
          <Button icon={<ReloadOutlined />} loading={loading} onClick={refresh}>刷新状态</Button>
          <Button type="primary" icon={<CloudDownloadOutlined />} loading={preloading} onClick={preload}>下载并预热</Button>
        </Space>
      )}
    >
      <Space direction="vertical" style={{ width: '100%' }} size={16}>
        <Space size={12} wrap>
          <Card size="small"><Statistic title="依赖状态" value={status?.available ? '已安装' : '未安装'} valueStyle={{ fontSize: 18, color: status?.available ? '#52c41a' : '#ff4d4f' }} /></Card>
          <Card size="small"><Statistic title="当前加载" value={status?.loaded ? '已加载' : '未加载'} valueStyle={{ fontSize: 18, color: status?.loaded ? '#1890ff' : '#8c8c8c' }} /></Card>
          <Card size="small"><Statistic title="当前模型" value={(status?.model ?? modelName).split('/').pop()} valueStyle={{ fontSize: 18 }} /></Card>
        </Space>
        <Descriptions bordered size="small" column={2}>
          <Descriptions.Item label="后端引擎">{status?.backend ?? '-'}</Descriptions.Item>
          <Descriptions.Item label="模型仓库">{status?.model ?? modelName}</Descriptions.Item>
        </Descriptions>
        <Alert
          type={status?.available ? 'info' : 'warning'}
          showIcon
          message={status?.available ? '模型下载说明' : '当前环境未安装 chronos-forecasting'}
          description={status?.available ? '首次预热会下载并缓存模型，后续时序任务和部署会复用本地缓存。' : '请先安装依赖，再使用模型下载与预热功能。'}
        />
      </Space>
    </Card>
  )
}

function DeploymentPanel({ stats, loading, data, columns, refresh, onNew }) {
  return (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      <Space size={12} wrap>
        <Card size="small"><Statistic title="部署总数" value={stats.total} /></Card>
        <Card size="small"><Statistic title="运行中" value={stats.active} valueStyle={{ color: '#52c41a' }} /></Card>
        <Card size="small"><Statistic title="累计调用" value={stats.calls} valueStyle={{ color: '#722ed1' }} /></Card>
      </Space>
      <Card
        className="ts-panel-card"
        extra={<Space><Button size="small" icon={<ReloadOutlined />} onClick={refresh}>刷新</Button><Button size="small" type="primary" icon={<PlusOutlined />} onClick={onNew}>新增部署</Button></Space>}
      >
        <Table
          rowKey={(row) => row.deployment_id ?? row.id}
          columns={columns}
          dataSource={data}
          loading={loading}
          locale={{ emptyText: <Empty description="暂无部署记录" /> }}
          scroll={{ x: 920 }}
          pagination={{ pageSize: 8, showSizeChanger: false, showTotal: (total) => `共 ${total} 条` }}
        />
      </Card>
    </Space>
  )
}

export default function ModelDeploy() {
  const [mlList, setMlList] = useState([])
  const [dlList, setDlList] = useState([])
  const [tsList, setTsList] = useState([])
  const [loading, setLoading] = useState({ ml: false, dl: false, ts: false })
  const [viewer, setViewer] = useState({ open: false, kind: 'ml', record: null, tab: 'overview' })
  const [tester, setTester] = useState({ ml: { input: '[\n  {}\n]', result: null, loading: false }, dl: { input: '[\n  {}\n]', result: null, loading: false }, ts: { input: '{\n  "dataset_id": "",\n  "value_column": "",\n  "time_column": null,\n  "horizon": 24,\n  "frequency": "high"\n}', result: null, loading: false } })
  const [modals, setModals] = useState({ ml: false, dl: false, ts: false })
  const [createLoading, setCreateLoading] = useState({ ml: false, dl: false, ts: false })
  const [mlTasks, setMlTasks] = useState([])
  const [dlTasks, setDlTasks] = useState([])
  const [tsStatus, setTsStatus] = useState(null)
  const [tsStatusLoading, setTsStatusLoading] = useState(false)
  const [tsPreloading, setTsPreloading] = useState(false)
  const [tsModelName, setTsModelName] = useState('amazon/chronos-t5-small')

  const patchTester = (kind, patch) => setTester((s) => ({ ...s, [kind]: { ...s[kind], ...patch } }))
  const openViewer = (kind, record, tab = 'overview') => { patchTester(kind, { result: null }); setViewer({ open: true, kind, record, tab }) }

  const fetchTsStatus = async () => {
    setTsStatusLoading(true)
    try { const res = await tsApi.modelStatus(); setTsStatus(res); if (res?.model) setTsModelName(res.model) }
    catch (err) { message.error(err?.response?.data?.detail ?? '获取 TimesFM 模型状态失败') }
    finally { setTsStatusLoading(false) }
  }

  const fetchMl = async () => {
    setLoading((s) => ({ ...s, ml: true }))
    try { const res = await deployApi.listDeployments(); setMlList(res.deployments ?? []) }
    catch { message.error('获取机器学习部署失败') }
    finally { setLoading((s) => ({ ...s, ml: false })) }
  }
  const fetchDl = async () => {
    setLoading((s) => ({ ...s, dl: true }))
    try { const res = await dlApi.listDeployments(); setDlList(res.deployments ?? []) }
    catch { message.error('获取深度学习部署失败') }
    finally { setLoading((s) => ({ ...s, dl: false })) }
  }
  const fetchTs = async () => {
    setLoading((s) => ({ ...s, ts: true }))
    try { const res = await tsApi.listDeployments({ page: 1, page_size: 100 }); setTsList(res.items ?? []) }
    catch { message.error('获取 TimesFM 部署失败') }
    finally { setLoading((s) => ({ ...s, ts: false })) }
  }

  useEffect(() => { void fetchMl(); void fetchDl(); void fetchTs(); void fetchTsStatus() }, [])
  useEffect(() => { if (viewer.open && viewer.tab === 'testing' && viewer.record) void preparePayload(viewer.kind, viewer.record) }, [viewer])

  const preparePayload = async (kind, record) => {
    try {
      if (kind === 'ml' && record.task_id) {
        const detail = await modelApi.getModelDetail(record.task_id)
        const preview = await dataApi.previewDataset(detail.dataset?.id ?? detail.dataset_id)
        if (preview?.rows?.[0] && detail.target_column) { const row = { ...preview.rows[0] }; delete row[detail.target_column]; patchTester('ml', { input: JSON.stringify([row], null, 2) }) }
      }
      if (kind === 'dl' && record.dl_task_id) {
        const task = await dlApi.getStatus(record.dl_task_id)
        const preview = await dataApi.previewDataset(task.dataset_id)
        if (preview?.rows?.[0] && task.target_column) { const row = { ...preview.rows[0] }; delete row[task.target_column]; patchTester('dl', { input: JSON.stringify([row], null, 2) }) }
      }
      if (kind === 'ts') {
        const dsRes = await dataApi.listDatasets({ page: 1, page_size: 20 })
        const ds = (dsRes.items ?? dsRes.datasets ?? [])[0]
        if (!ds) return
        let valueColumn = 'value'
        let timeColumn = null
        try {
          const preview = await dataApi.previewDataset(ds.id)
          const cols = preview?.rows?.[0] ? Object.keys(preview.rows[0]) : []
          timeColumn = cols.find((c) => ['date', 'time', 'datetime', 'timestamp'].includes(c.toLowerCase())) ?? null
          valueColumn = cols.find((c) => c !== timeColumn) ?? cols[0] ?? valueColumn
        } catch {}
        patchTester('ts', { input: JSON.stringify({ dataset_id: ds.id, value_column: valueColumn, time_column: timeColumn, horizon: 24, frequency: 'high' }, null, 2) })
      }
    } catch {}
  }

  const toggleStatus = async (kind, record) => {
    const next = record.status === 'active' ? 'paused' : 'active'
    try {
      if (kind === 'ml') await deployApi.updateStatus(record.deployment_id, next)
      else if (kind === 'dl') await dlApi.toggleDeployment(record.id, next)
      else await tsApi.updateDeploymentStatus(record.deployment_id, next)
      message.success(next === 'active' ? '部署已启用' : '部署已暂停')
      if (kind === 'ml') void fetchMl(); else if (kind === 'dl') void fetchDl(); else void fetchTs()
    } catch (err) { message.error(err?.response?.data?.detail ?? '切换部署状态失败') }
  }

  const removeDeployment = async (kind, record) => {
    try {
      if (kind === 'ml') await deployApi.deleteDeployment(record.deployment_id)
      else if (kind === 'dl') await dlApi.deleteDeployment(record.id)
      else await tsApi.deleteDeployment(record.deployment_id)
      message.success('部署已删除')
      if (kind === 'ml') void fetchMl(); else if (kind === 'dl') void fetchDl(); else void fetchTs()
      if ((viewer.record?.deployment_id ?? viewer.record?.id) === (record.deployment_id ?? record.id)) setViewer({ open: false, kind: 'ml', record: null, tab: 'overview' })
    } catch (err) { message.error(err?.response?.data?.detail ?? '删除部署失败') }
  }

  const handleTest = async () => {
    if (!viewer.record) return
    const kind = viewer.kind
    patchTester(kind, { loading: true, result: null })
    try {
      let result
      if (kind === 'ml') { const rows = JSON.parse(tester[kind].input); result = await deployApi.predict(viewer.record.deployment_id, { rows: Array.isArray(rows) ? rows : [rows], include_probabilities: true }) }
      else if (kind === 'dl') { const rows = JSON.parse(tester[kind].input); result = await dlApi.predictDeployment(viewer.record.id, { rows: Array.isArray(rows) ? rows : [rows] }) }
      else { result = await tsApi.predictDeployment(viewer.record.deployment_id, JSON.parse(tester[kind].input)) }
      patchTester(kind, { result })
    } catch (err) { message.error(err?.response?.data?.detail ?? err?.message ?? '在线测试失败') }
    finally { patchTester(kind, { loading: false }) }
  }

  const preloadTsModel = async () => {
    setTsPreloading(true)
    try { const res = await tsApi.preloadModel(tsModelName); message.success(res?.message ?? '已开始后台下载并预热模型'); await fetchTsStatus() }
    catch (err) { message.error(err?.response?.data?.detail ?? '触发模型下载失败') }
    finally { setTsPreloading(false) }
  }

  const openCreate = async (kind) => {
    if (kind === 'ml' && !mlTasks.length) {
      const res = await trainingApi.listTasks({ page: 1, page_size: 100, status: 'SUCCESS' }).catch(() => ({ items: [] }))
      setMlTasks(res.items ?? [])
    }
    if (kind === 'dl' && !dlTasks.length) {
      const res = await dlApi.listTasks({ page: 1, page_size: 100, status: 'SUCCESS' }).catch(() => ({ items: [] }))
      setDlTasks(res.items ?? [])
    }
    setModals((s) => ({ ...s, [kind]: true }))
  }

  const submitCreate = async (kind, values) => {
    setCreateLoading((s) => ({ ...s, [kind]: true }))
    try {
      if (kind === 'ml') await deployApi.createDeployment(values.task_id, { name: values.name, description: values.description ?? '' })
      else if (kind === 'dl') await dlApi.createDeployment(values.task_id, { name: values.name, description: values.description ?? '' })
      else await tsApi.createDeployment(values)
      message.success('部署已创建')
      setModals((s) => ({ ...s, [kind]: false }))
      if (kind === 'ml') void fetchMl(); else if (kind === 'dl') void fetchDl(); else { void fetchTs(); void fetchTsStatus() }
    } catch (err) { message.error(err?.response?.data?.detail ?? '创建部署失败') }
    finally { setCreateLoading((s) => ({ ...s, [kind]: false })) }
  }

  const predictUrl = (kind, record) => !record ? '' : kind === 'ml' ? (record.endpoints?.predict ?? '') : kind === 'dl' ? `${api.defaults.baseURL}/dl/deployments/${record.id}/predict` : (record.predict_url ?? '')

  const detailItems = (kind, record) => {
    if (!record) return []
    const meta = statusMeta(record.status)
    const items = [
      { key: 'status', label: '状态', children: <Badge status={meta.badge} text={meta.text} /> },
      { key: 'created', label: '创建时间', children: formatDateTime(record.created_at) },
      { key: 'calls', label: '调用次数', children: record.request_count ?? 0 },
      { key: 'desc', label: '说明', children: record.description || '无' },
    ]
    if (kind === 'ml') items.push({ key: 'task', label: '训练任务 ID', children: <Text copyable>{record.task_id ?? '-'}</Text> })
    if (kind === 'dl') items.push({ key: 'task', label: '训练任务 ID', children: <Text copyable>{record.dl_task_id ?? '-'}</Text> })
    if (kind === 'ts') items.push({ key: 'backend', label: '基础模型', children: <Tag color="blue">{record.backend_label?.split('/').pop() ?? '-'}</Tag> })
    return items
  }

  const columns = (kind) => [
    { title: '部署名称', dataIndex: 'name', render: (value, record) => <Space direction="vertical" size={2}><Button type="link" style={{ padding: 0, fontWeight: 600 }} onClick={() => openViewer(kind, record, 'overview')}>{value}</Button>{kind === 'ts' && record.backend_label ? <Tag color="blue" style={{ fontSize: 10 }}>{record.backend_label.split('/').pop()}</Tag> : null}</Space> },
    { title: '状态', dataIndex: 'status', width: 110, render: (value) => { const meta = statusMeta(value); return <Badge status={meta.badge} text={<Text style={{ color: meta.color }}>{meta.text}</Text>} /> } },
    { title: '调用次数', dataIndex: 'request_count', width: 100, render: (value) => value ?? 0 },
    { title: '创建时间', dataIndex: 'created_at', width: 180, render: (value) => <Text style={{ fontSize: 12 }}>{formatDateTime(value)}</Text> },
    { title: '操作', key: 'actions', width: 210, render: (_, record) => <Space size={4}><Button size="small" icon={<EyeOutlined />} onClick={() => openViewer(kind, record, 'overview')}>详情</Button><Button size="small" icon={<ThunderboltOutlined />} onClick={() => openViewer(kind, record, 'testing')}>测试</Button><Button size="small" icon={record.status === 'active' ? <PauseCircleOutlined /> : <PlayCircleOutlined />} onClick={() => void toggleStatus(kind, record)} /><Popconfirm title="确认删除这个部署吗" okButtonProps={{ danger: true }} onConfirm={() => void removeDeployment(kind, record)}><Button size="small" danger icon={<DeleteOutlined />} /></Popconfirm></Space> },
  ]

  const stats = (list) => ({ total: list.length, active: list.filter((x) => x.status === 'active').length, calls: list.reduce((sum, x) => sum + (x.request_count ?? 0), 0) })
  const mlStats = useMemo(() => stats(mlList), [mlList])
  const dlStats = useMemo(() => stats(dlList), [dlList])
  const tsStats = useMemo(() => stats(tsList), [tsList])

  return (
    <Space direction="vertical" size={20} style={{ width: '100%' }}>
      <div className="ts-page-hero">
        <Space direction="vertical" size={4}>
          <Title level={2} style={{ margin: 0 }}><Space><CloudServerOutlined />模型部署</Space></Title>
          <Text type="secondary">统一查看机器学习、深度学习和 TimesFM 的部署实例，支持服务地址查看、状态切换和在线测试。</Text>
        </Space>
      </div>

      <Tabs
        defaultActiveKey="ml"
        items={[
          { key: 'ml', label: '机器学习', children: <DeploymentPanel stats={mlStats} loading={loading.ml} data={mlList} columns={columns('ml')} refresh={() => void fetchMl()} onNew={() => void openCreate('ml')} /> },
          { key: 'dl', label: '深度学习', children: <DeploymentPanel stats={dlStats} loading={loading.dl} data={dlList} columns={columns('dl')} refresh={() => void fetchDl()} onNew={() => void openCreate('dl')} /> },
          { key: 'timesfm', label: 'TimesFM', children: <Space direction="vertical" size={16} style={{ width: '100%' }}><TimesFmCard status={tsStatus} loading={tsStatusLoading} preloading={tsPreloading} modelName={tsModelName} setModelName={setTsModelName} refresh={() => void fetchTsStatus()} preload={() => void preloadTsModel()} /><DeploymentPanel stats={tsStats} loading={loading.ts} data={tsList} columns={columns('ts')} refresh={() => void fetchTs()} onNew={() => void openCreate('ts')} /></Space> },
        ]}
      />

      <Drawer title={viewer.record ? <Space><CloudServerOutlined />{viewer.record.name}</Space> : '部署详情'} open={viewer.open} onClose={() => setViewer({ open: false, kind: 'ml', record: null, tab: 'overview' })} width={720} destroyOnClose>
        {!viewer.record ? <Empty description="请选择一个部署实例" /> : (
          <Tabs
            activeKey={viewer.tab}
            destroyInactiveTabPane
            onChange={(tab) => setViewer((s) => ({ ...s, tab }))}
            items={[
              { key: 'overview', label: '概览', children: <Descriptions bordered column={1} size="small" items={detailItems(viewer.kind, viewer.record)} /> },
              { key: 'endpoints', label: '服务地址', children: <CopyField label="预测 URL" value={predictUrl(viewer.kind, viewer.record)} /> },
              { key: 'testing', label: '在线测试', children: <Space direction="vertical" size={16} style={{ width: '100%' }}><Alert type="info" showIcon message="在线测试说明" description={viewer.kind === 'ts' ? 'TimesFM 请求需要 dataset_id、value_column、time_column、horizon 和 frequency。' : 'ML 和 DL 请求需要 JSON 数组，每一项代表一条待预测样本。'} /><Input.TextArea rows={10} value={tester[viewer.kind].input} onChange={(e) => patchTester(viewer.kind, { input: e.target.value })} style={{ fontFamily: 'Consolas, Monaco, monospace', fontSize: 12 }} /><div className="ts-drawer-footer"><Button type="primary" icon={<ThunderboltOutlined />} loading={tester[viewer.kind].loading} disabled={viewer.record.status !== 'active'} onClick={() => void handleTest()}>发送测试请求</Button></div>{tester[viewer.kind].result ? <pre className="ts-json-block">{JSON.stringify(tester[viewer.kind].result, null, 2)}</pre> : null}</Space> },
            ]}
          />
        )}
      </Drawer>

      <CreateDeploymentModal
        open={modals.ml}
        onClose={() => setModals((s) => ({ ...s, ml: false }))}
        onSubmit={(values) => submitCreate('ml', values)}
        loading={createLoading.ml}
        title="新增机器学习部署"
        fields={<><Form.Item label="训练任务" name="task_id" rules={[{ required: true, message: '请选择训练完成的任务' }]}><Select showSearch optionFilterProp="label" placeholder="选择一个训练完成的机器学习任务" options={mlTasks.map((task) => ({ value: task.id ?? task.task_id, label: `${task.task_name ?? task.name ?? task.id?.slice(0, 8)} / ${task.model_type ?? ''}` }))} /></Form.Item><Form.Item label="部署名称" name="name" rules={[{ required: true, message: '请输入部署名称' }]}><Input placeholder="例如：XGBoost 生产部署 v1" /></Form.Item><Form.Item label="部署说明" name="description"><Input.TextArea rows={3} placeholder="记录部署用途、调用方或版本说明" /></Form.Item></>}
      />

      <CreateDeploymentModal
        open={modals.dl}
        onClose={() => setModals((s) => ({ ...s, dl: false }))}
        onSubmit={(values) => submitCreate('dl', values)}
        loading={createLoading.dl}
        title="新增深度学习部署"
        fields={<><Form.Item label="训练任务" name="task_id" rules={[{ required: true, message: '请选择训练完成的任务' }]}><Select showSearch optionFilterProp="label" placeholder="选择一个训练完成的深度学习任务" options={dlTasks.map((task) => ({ value: task.id ?? task.task_id, label: `${task.task_name ?? task.name ?? task.id?.slice(0, 8)} / ${task.model_type ?? ''}` }))} /></Form.Item><Form.Item label="部署名称" name="name" rules={[{ required: true, message: '请输入部署名称' }]}><Input placeholder="例如：Transformer 线上部署 v1" /></Form.Item><Form.Item label="部署说明" name="description"><Input.TextArea rows={3} placeholder="记录部署用途、调用方或版本说明" /></Form.Item></>}
      />

      <CreateDeploymentModal
        open={modals.ts}
        onClose={() => setModals((s) => ({ ...s, ts: false }))}
        onSubmit={(values) => submitCreate('ts', values)}
        loading={createLoading.ts}
        title="新增 TimesFM 部署"
        fields={<><Form.Item label="部署名称" name="name" rules={[{ required: true, message: '请输入部署名称' }]}><Input placeholder="例如：Chronos 生产服务 A" /></Form.Item><Form.Item label="基础模型" name="backend_label" initialValue="amazon/chronos-t5-small"><Select options={[{ value: 'amazon/chronos-t5-tiny', label: 'Chronos T5-Tiny' }, { value: 'amazon/chronos-t5-small', label: 'Chronos T5-Small' }, { value: 'amazon/chronos-t5-base', label: 'Chronos T5-Base' }, { value: 'amazon/chronos-t5-large', label: 'Chronos T5-Large' }]} /></Form.Item><Form.Item label="部署说明" name="description"><Input.TextArea rows={3} placeholder="记录服务用途、数据范围或调用方" /></Form.Item></>}
      />
    </Space>
  )
}
