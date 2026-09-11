import { useActiveEffect } from '../hooks/useActiveEffect'
import React, { useCallback, useState } from 'react'
import {
  Badge,
  Button,
  Card,
  Col,
  Descriptions,
  Divider,
  Form,
  Input,
  Modal,
  Row,
  Select,
  Slider,
  Space,
  Spin,
  Switch,
  Tag,
  Typography,
  message,
} from '../ui'
import { ApiOutlined, PlusOutlined, ReloadOutlined, SettingOutlined, TagOutlined } from '@ant-design/icons'
import { modelApi, systemApi } from '../services/api'
import { REFRESH_SECONDS_RANGE, resetPrefs, setPrefs, useAppSettings } from '../hooks/useAppSettings'

const { Title, Text } = Typography
const { Option } = Select

// ── Dimension config ──────────────────────────────────────────────────────────
const DIMENSIONS = [
  { key: '类别',    color: 'blue',    desc: '模型的算法类别（分类、回归等）' },
  { key: '规模',    color: 'green',   desc: '模型的参数量与复杂度' },
  { key: '目的',    color: 'orange',  desc: '模型的使用场景与意图' },
  { key: '领域',    color: 'purple',  desc: '模型所属的业务领域' },
  { key: '数据类型', color: 'cyan',   desc: '训练数据的类型' },
  { key: '其他',    color: 'default', desc: '未分类标签' },
]

// ── Tag Library Management ────────────────────────────────────────────────────
function TagLibrarySection() {
  const [grouped, setGrouped] = useState({})
  const [allTags, setAllTags] = useState([])
  const [loading, setLoading] = useState(false)
  const [addModal, setAddModal] = useState(false)
  const [addForm] = Form.useForm()
  const [adding, setAdding] = useState(false)

  useActiveEffect(() => { void fetchTags() }, [])

  async function fetchTags() {
    setLoading(true)
    try {
      const res = await modelApi.listTags()
      setAllTags(res.tags ?? [])
      setGrouped(res.grouped ?? {})
    } catch {
      message.error('加载标签库失败')
    } finally {
      setLoading(false)
    }
  }

  async function handleAddTag(values) {
    const name = values.name?.trim()
    if (!name) return
    if (allTags.some(t => t.name === name)) { message.warning('标签已存在'); return }
    setAdding(true)
    try {
      const dim = DIMENSIONS.find(d => d.key === values.dimension)
      await modelApi.createTag(name, values.dimension || null, dim?.color || null)
      message.success(`已添加标签「${name}」`)
      setAddModal(false)
      addForm.resetFields()
      void fetchTags()
    } catch {
      message.error('添加失败')
    } finally {
      setAdding(false)
    }
  }

  async function handleDeleteTag(name) {
    try {
      const res = await modelApi.deleteTag(name)
      setAllTags(res.tags ?? [])
      const g = {}
      for (const t of res.tags ?? []) {
        const dim = t.dimension || '其他'
        ;(g[dim] = g[dim] ?? []).push(t)
      }
      setGrouped(g)
      message.success(`已删除标签「${name}」`)
    } catch {
      message.error('删除失败')
    }
  }

  const dimOrder = DIMENSIONS.map(d => d.key)
  const sortedDims = Object.keys(grouped).sort(
    (a, b) => dimOrder.indexOf(a) - dimOrder.indexOf(b)
  )

  return (
    <Card
      title={<Space><TagOutlined style={{ color: '#1a8dff' }} /><span>标签库管理</span></Space>}
      extra={
        <Space>
          <Button size="small" onClick={() => void fetchTags()}>刷新</Button>
          <Button size="small" type="primary" icon={<PlusOutlined />} onClick={() => setAddModal(true)}>
            新建标签
          </Button>
        </Space>
      }
    >
      <Text type="secondary" style={{ display: 'block', marginBottom: 16 }}>
        标签按维度分类管理，可在「模型管理」页面为模型打标签时复用。
      </Text>

      {loading ? (
        <div style={{ textAlign: 'center', padding: 24 }}><Spin /></div>
      ) : (
        <Row gutter={[16, 16]}>
          {DIMENSIONS.filter(d => grouped[d.key]?.length > 0).map(dim => {
            const tags = grouped[dim.key] ?? []
            return (
              <Col xs={24} sm={12} xl={8} key={dim.key}>
                <Card
                  size="small"
                  title={
                    <Space size={6}>
                      <Tag color={dim.color} style={{ margin: 0 }}>{dim.key}</Tag>
                      <Text type="secondary" style={{ fontSize: 11 }}>{dim.desc}</Text>
                    </Space>
                  }
                  bodyStyle={{ padding: '8px 12px' }}
                >
                  <Space wrap size={[6, 6]}>
                    {tags.map(tag => (
                      <Tag
                        key={tag.name}
                        color={tag.color ?? dim.color}
                        closable
                        onClose={e => { e.preventDefault(); void handleDeleteTag(tag.name) }}

                      >
                        {tag.name}
                      </Tag>
                    ))}
                  </Space>
                </Card>
              </Col>
            )
          })}
        </Row>
      )}

      {allTags.length === 0 && !loading && (
        <Text type="secondary">暂无标签，点击「新建标签」添加。</Text>
      )}

      <Divider style={{ margin: '12px 0 4px' }} />
      <Text type="secondary" style={{ fontSize: 12 }}>
        共 {allTags.length} 个标签，{sortedDims.length} 个维度 · 点击标签右侧 × 可删除
      </Text>

      <Modal
        title={<Space><PlusOutlined />新建标签</Space>}
        open={addModal}
        onCancel={() => { setAddModal(false); addForm.resetFields() }}
        onOk={() => addForm.submit()}
        okText="添加"
        confirmLoading={adding}
        destroyOnHidden
        width={400}
      >
        <Form form={addForm} layout="vertical" onFinish={handleAddTag} style={{ marginTop: 16 }}>
          <Form.Item
            name="name"
            label="标签名称"
            rules={[{ required: true, message: '请输入标签名称' }]}
          >
            <Input placeholder="例：高精度、季节性数据" maxLength={30} />
          </Form.Item>
          <Form.Item name="dimension" label="所属维度">
            <Select placeholder="选择维度（可选）" allowClear>
              {DIMENSIONS.filter(d => d.key !== '其他').map(d => (
                <Option key={d.key} value={d.key}>
                  <Tag color={d.color} style={{ marginRight: 6 }}>{d.key}</Tag>
                  {d.desc}
                </Option>
              ))}
            </Select>
          </Form.Item>
        </Form>
      </Modal>
    </Card>
  )
}

// ── 服务连接（只读，反映真实解析结果，不是可改的设置项）─────────────────────
function ConnectionSection() {
  const [state, setState] = useState({ loading: true, health: null, error: null, latency: null })

  const origin = typeof window === 'undefined' ? '' : window.location.origin
  const wsOrigin = origin.replace(/^http/, 'ws')

  const check = useCallback(async () => {
    setState(prev => ({ ...prev, loading: true }))
    const started = performance.now()
    try {
      const health = await systemApi.health()
      setState({ loading: false, health, error: null, latency: Math.round(performance.now() - started) })
    } catch (err) {
      setState({
        loading: false,
        health: null,
        latency: null,
        error: err?.message ?? '无法连接后端',
      })
    }
  }, [])

  useActiveEffect(() => { void check() }, [check])

  const { health, error, loading, latency } = state
  const uploadMb = health?.max_upload_size_mb

  return (
    <Card
      title={<Space><ApiOutlined style={{ color: '#1a8dff' }} /><span>服务连接</span></Space>}
      extra={<Button size="small" onClick={() => void check()} loading={loading} icon={<ReloadOutlined />}>重新检测</Button>}
    >
      <Text type="secondary" style={{ display: 'block', marginBottom: 16 }}>
        前端始终按当前访问的地址回连后端，没有可填的服务器地址——填了反而会连错。
        这里显示的是实际生效的结果。
      </Text>

      <Descriptions bordered size="small" column={{ xs: 1, md: 2 }}>
        <Descriptions.Item label="接口地址">
          <Text code copyable>{`${origin}/api`}</Text>
        </Descriptions.Item>
        <Descriptions.Item label="WebSocket 地址">
          <Text code copyable>{`${wsOrigin}/ws`}</Text>
        </Descriptions.Item>
        <Descriptions.Item label="后端状态">
          {loading ? <Spin size="small" /> : error
            ? <Badge status="error" text={<Text type="danger">{error}</Text>} />
            : <Badge status="success" text={`正常${latency == null ? '' : ` · ${latency} ms`}`} />}
        </Descriptions.Item>
        <Descriptions.Item label="后端版本">
          {health?.version ?? '—'}
          {health?.environment ? <Tag style={{ marginLeft: 8 }}>{health.environment}</Tag> : null}
        </Descriptions.Item>
        <Descriptions.Item label="上传大小上限" span={2}>
          {uploadMb == null ? '—' : `${uploadMb} MB`}
          <Text type="secondary" style={{ marginLeft: 8, fontSize: 12 }}>
            由后端 MAX_UPLOAD_SIZE 决定，前端改不了
          </Text>
        </Descriptions.Item>
      </Descriptions>
    </Card>
  )
}

// ── 界面偏好（写 localStorage，页面轮询实时读取）──────────────────────────────
function PreferenceSection() {
  const prefs = useAppSettings()

  return (
    <Card title={<Space><SettingOutlined style={{ color: '#1a8dff' }} /><span>界面偏好</span></Space>}>
      <Text type="secondary" style={{ display: 'block', marginBottom: 20 }}>
        存在本机浏览器里，改完立即生效，不需要保存，也不影响其他人。
      </Text>

      <div className="settings-field">
        <div className="settings-field-label">
          <Text strong>自动刷新</Text>
          <Switch
            checked={prefs.autoRefresh}
            onChange={(checked) => setPrefs({ autoRefresh: checked })}
          />
        </div>
        <Text type="secondary" style={{ fontSize: 12 }}>
          关掉之后，任务详情、训练过程、时序任务列表等页面只在你手动刷新时拉取数据。
        </Text>
      </div>

      <Divider style={{ margin: '20px 0' }} />

      <div className="settings-field">
        <div className="settings-field-label">
          <Text strong>刷新间隔</Text>
          <Text style={{ fontVariantNumeric: 'tabular-nums' }}>{prefs.refreshSeconds} 秒</Text>
        </div>
        <Text type="secondary" style={{ fontSize: 12 }}>
          训练中想看得实时一点就调小；任务多、网络慢时调大可以少发一些请求。
        </Text>
        <Slider
          min={REFRESH_SECONDS_RANGE.min}
          max={REFRESH_SECONDS_RANGE.max}
          step={1}
          disabled={!prefs.autoRefresh}
          value={prefs.refreshSeconds}
          onChange={(value) => setPrefs({ refreshSeconds: value })}
          marks={{ 2: '2s', 5: '5s', 15: '15s', 30: '30s', 60: '60s' }}
          style={{ marginTop: 8 }}
        />
      </div>

      <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 28 }}>
        <Button onClick={() => { resetPrefs(); message.success('已恢复默认') }}>恢复默认</Button>
      </div>
    </Card>
  )
}

// ── Main Settings Page ────────────────────────────────────────────────────────
const Settings = () => (
  <Space direction="vertical" size={20} style={{ width: '100%' }}>
    <Title level={2} style={{ margin: 0 }}>
      <Space>
        <SettingOutlined />
        系统设置
      </Space>
    </Title>

    <ConnectionSection />
    <PreferenceSection />
    <TagLibrarySection />
  </Space>
)

export default Settings
