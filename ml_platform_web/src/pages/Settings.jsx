import React, { useState } from 'react'
import { 
  Card, 
  Form, 
  Input, 
  Button, 
  Switch, 
  Select, 
  Slider, 
  Typography, 
  Space, 
  message,
  Divider
} from 'antd'
import { SettingOutlined, SaveOutlined } from '@ant-design/icons'

const { Title, Text } = Typography
const { Option } = Select

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
    refreshInterval: 5
  }

  const handleSubmit = () => {
    form.validateFields()
      .then(values => {
        setLoading(true)
        // 模拟保存设置
        setTimeout(() => {
          setLoading(false)
          message.success('设置保存成功')
        }, 1000)
      })
      .catch(info => {
        console.log('验证失败:', info)
      })
  }

  return (
    <div>
      <Title level={2}>系统设置</Title>
      
      <Card hoverable>
        <Form form={form} layout="vertical" initialValues={defaultSettings}>
          <Title level={4}>API设置</Title>
          <Form.Item 
            name="apiBaseUrl" 
            label="API基础地址" 
            rules={[{ required: true, message: '请输入API基础地址' }]}
          >
            <Input placeholder="请输入API基础地址" />
          </Form.Item>

          <Form.Item 
            name="websocketUrl" 
            label="WebSocket地址" 
            rules={[{ required: true, message: '请输入WebSocket地址' }]}
          >
            <Input placeholder="请输入WebSocket地址" />
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
          <Form.Item 
            name="autoSave" 
            label="自动保存" 
            valuePropName="checked"
          >
            <Switch />
          </Form.Item>

          <Form.Item 
            name="notification" 
            label="通知提醒" 
            valuePropName="checked"
          >
            <Switch />
          </Form.Item>

          <Form.Item 
            name="maxUploadSize" 
            label="最大上传大小 (MB)"
          >
            <Slider min={10} max={500} step={10} />
          </Form.Item>

          <Form.Item 
            name="refreshInterval" 
            label="刷新间隔 (秒)"
          >
            <Slider min={1} max={30} step={1} />
          </Form.Item>

          <div className="mt-6 flex justify-end">
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
    </div>
  )
}

export default Settings