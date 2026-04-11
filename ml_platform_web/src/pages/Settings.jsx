import React, { useEffect, useRef, useState } from 'react'
import {
  Button,
  Card,
  Divider,
  Form,
  Input,
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

// ── Tag Library Management ────────────────────────────────────────────────────
function TagLibrarySection() {
  const [tags, setTags] = useState([])
  const [loading, setLoading] = useState(false)
  const [inputVisible, setInputVisible] = useState(false)
  const [inputValue, setInputValue] = useState('')
  const [adding, setAdding] = useState(false)
  const inputRef = useRef(null)

  useEffect(() => { void fetchTags() }, [])

  useEffect(() => {
    if (inputVisible) {
      setTimeout(() => inputRef.current?.focus(), 100)
    }
  }, [inputVisible])

  async function fetchTags() {
    setLoading(true)
    try {
      const res = await modelApi.listTags()
      setTags(res.tags ?? [])
    } catch {
      message.error('加载标签库失败')
    } finally {
      setLoading(false)
    }
  }

  async function handleAddTag() {
    const trimmed = inputValue.trim()
    if (!trimmed) { setInputVisible(false); setInputValue(''); return }
    if (tags.includes(trimmed)) { message.warning('标签已存在'); return }
    setAdding(true)
    try {
      const res = await modelApi.syncTags([trimmed])
      setTags(res.tags ?? [])
      message.success(`已添加标签「${trimmed}」`)
    } catch {
      message.error('添加失败')
    } finally {
      setAdding(false)
      setInputVisible(false)
      setInputValue('')
    }
  }

  async function handleDeleteTag(name) {
    try {
      const res = await modelApi.deleteTag(name)
      setTags(res.tags ?? [])
      message.success(`已删除标签「${name}」`)
    } catch {
      message.error('删除失败')
    }
  }

  return (
    <Card
      title={
        <Space>
          <TagOutlined style={{ color: '#1890ff' }} />
          <span>标签库管理</span>
        </Space>
      }
      extra={
        <Button size="small" onClick={() => void fetchTags()}>
          刷新
        </Button>
      }
    >
      <Text type="secondary" style={{ display: 'block', marginBottom: 16 }}>
        在此管理模型标签库。标签可在「模型管理」页面为模型打标签时复用。
      </Text>

      {loading ? (
        <div style={{ textAlign: 'center', padding: 24 }}>
          <Spin />
        </div>
      ) : (
        <Space wrap size={[8, 8]} style={{ marginBottom: 16 }}>
          {tags.map((tag) => (
            <Tag
              key={tag}
              color="geekblue"
              closable
              onClose={(e) => { e.preventDefault(); void handleDeleteTag(tag) }}
              style={{ fontSize: 13, padding: '2px 8px' }}
            >
              {tag}
            </Tag>
          ))}

          {inputVisible ? (
            <Input
              ref={inputRef}
              size="small"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onBlur={() => void handleAddTag()}
              onPressEnter={() => void handleAddTag()}
              style={{ width: 120 }}
              placeholder="输入后回车"
              disabled={adding}
            />
          ) : (
            <Tag
              onClick={() => setInputVisible(true)}
              style={{ cursor: 'pointer', borderStyle: 'dashed', background: '#fff' }}
            >
              <PlusOutlined /> 新建标签
            </Tag>
          )}
        </Space>
      )}

      {tags.length === 0 && !loading && (
        <Text type="secondary">暂无标签，点击「新建标签」添加。</Text>
      )}

      <Divider style={{ margin: '8px 0 0' }} />
      <Text type="secondary" style={{ fontSize: 12 }}>
        共 {tags.length} 个标签 · 点击标签右侧 × 可删除
      </Text>
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
