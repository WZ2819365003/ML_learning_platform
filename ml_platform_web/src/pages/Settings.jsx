import React, { useEffect, useState } from 'react'
import {
  Button,
  Card,
  Col,
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
} from 'antd'
import { PlusOutlined, SaveOutlined, SettingOutlined, TagOutlined } from '@ant-design/icons'
import { modelApi } from '../services/api'

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

  useEffect(() => { void fetchTags() }, [])

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
      title={<Space><TagOutlined style={{ color: '#1890ff' }} /><span>标签库管理</span></Space>}
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
                        style={{ fontSize: 13, padding: '2px 8px' }}
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

// ── Main Settings Page ────────────────────────────────────────────────────────
const Settings = () => {
  const [form] = Form.useForm()
  const [loading, setLoading] = useState(false)

  const defaultSettings = {
    apiBaseUrl: 'http://localhost:8000',
    websocketUrl: 'ws://localhost:8000',
    autoSave: true,
    theme: 'light',
    language: 'zh-CN',
    notification: true,
    maxUploadSize: 200,
    refreshInterval: 5,
  }

  const handleSubmit = () => {
    form.validateFields()
      .then(() => {
        setLoading(true)
        setTimeout(() => {
          setLoading(false)
          message.success('设置保存成功')
        }, 1000)
      })
      .catch((info) => {
        console.log('验证失败:', info)
      })
  }

  return (
    <Space direction="vertical" size={20} style={{ width: '100%' }}>
      <Title level={2} style={{ margin: 0 }}>
        <Space>
          <SettingOutlined />
          系统设置
        </Space>
      </Title>

      {/* ── General settings form ── */}
      <Card hoverable>
        <Form form={form} layout="vertical" initialValues={defaultSettings}>
          <Title level={4}>API 设置</Title>
          <Form.Item
            name="apiBaseUrl"
            label="API 基础地址"
            rules={[{ required: true, message: '请输入 API 基础地址' }]}
          >
            <Input placeholder="请输入 API 基础地址" />
          </Form.Item>

          <Form.Item
            name="websocketUrl"
            label="WebSocket 地址"
            rules={[{ required: true, message: '请输入 WebSocket 地址' }]}
          >
            <Input placeholder="请输入 WebSocket 地址" />
          </Form.Item>

          <Divider />

          <Title level={4}>界面设置</Title>
          <Form.Item
            name="theme"
            label="主题"
            rules={[{ required: true, message: '请选择主题' }]}
          >
            <Select placeholder="请选择主题">
              <Option value="light">浅色主题</Option>
              <Option value="dark">深色主题</Option>
              <Option value="system">跟随系统</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="language"
            label="语言"
            rules={[{ required: true, message: '请选择语言' }]}
          >
            <Select placeholder="请选择语言">
              <Option value="zh-CN">中文</Option>
              <Option value="en-US">英文</Option>
            </Select>
          </Form.Item>

          <Divider />

          <Title level={4}>功能设置</Title>
          <Form.Item name="autoSave" label="自动保存" valuePropName="checked">
            <Switch />
          </Form.Item>

          <Form.Item name="notification" label="通知提醒" valuePropName="checked">
            <Switch />
          </Form.Item>

          <Form.Item name="maxUploadSize" label="最大上传大小 (MB)">
            <Slider min={10} max={500} step={10} />
          </Form.Item>

          <Form.Item name="refreshInterval" label="刷新间隔 (秒)">
            <Slider min={1} max={30} step={1} />
          </Form.Item>

          <div style={{ marginTop: 24, display: 'flex', justifyContent: 'flex-end' }}>
            <Space>
              <Button onClick={() => form.resetFields()}>重置</Button>
              <Button
                type="primary"
                onClick={handleSubmit}
                loading={loading}
                icon={<SaveOutlined />}
              >
                保存设置
              </Button>
            </Space>
          </div>
        </Form>
      </Card>

      {/* ── Tag library management ── */}
      <TagLibrarySection />
    </Space>
  )
}

export default Settings
