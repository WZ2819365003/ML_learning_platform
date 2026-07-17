import React, { useState } from 'react'
import { Button, Form, Input, Typography, message } from 'antd'
import { LockOutlined, UserOutlined } from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'
import { authApi, setAuthToken } from '../services/api'

const { Text } = Typography

/* Same brand mark as the Sidebar so the login screen feels first-party. */
function BrandMark({ size = 48 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="36" height="36" rx="10" fill="url(#lgrad)" />
      <path d="M10 26 L18 10 L26 26" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" fill="none" opacity="0.9"/>
      <circle cx="18" cy="10" r="2.5" fill="#60a5fa" />
      <circle cx="10" cy="26" r="2" fill="white" opacity="0.7" />
      <circle cx="26" cy="26" r="2" fill="white" opacity="0.7" />
      <defs>
        <linearGradient id="lgrad" x1="0" y1="0" x2="36" y2="36" gradientUnits="userSpaceOnUse">
          <stop stopColor="#1d4ed8" />
          <stop offset="1" stopColor="#3b82f6" />
        </linearGradient>
      </defs>
    </svg>
  )
}

export default function Login() {
  const [submitting, setSubmitting] = useState(false)
  const navigate = useNavigate()

  const onFinish = async ({ username, password }) => {
    setSubmitting(true)
    try {
      const resp = await authApi.login(username, password)
      setAuthToken(resp.token)
      message.success('登录成功')
      navigate('/dashboard', { replace: true })
    } catch (err) {
      message.error(err?.response?.data?.detail || '登录失败，请稍后重试')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div style={{
      minHeight: '100dvh', display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: 'linear-gradient(160deg, #090f1e 0%, #0d1b38 55%, #0f2046 100%)',
      padding: 16,
    }}>
      <div style={{
        width: 380, maxWidth: '100%', background: '#fff', borderRadius: 16,
        boxShadow: '0 24px 64px rgba(9, 15, 30, 0.45)', padding: '40px 36px 28px',
      }}>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginBottom: 28 }}>
          <BrandMark size={52} />
          <div style={{ fontWeight: 700, fontSize: 20, marginTop: 14, letterSpacing: '-0.02em' }}>
            ML Platform
          </div>
          <Text type="secondary" style={{ fontSize: 12, marginTop: 4 }}>
            登录以访问建模工作台
          </Text>
        </div>

        <Form name="login" size="large" onFinish={onFinish} requiredMark={false}>
          <Form.Item name="username" rules={[{ required: true, message: '请输入用户名' }]}>
            <Input prefix={<UserOutlined style={{ color: '#94a3b8' }} />} placeholder="用户名" autoFocus />
          </Form.Item>
          <Form.Item name="password" rules={[{ required: true, message: '请输入密码' }]}>
            <Input.Password prefix={<LockOutlined style={{ color: '#94a3b8' }} />} placeholder="密码" />
          </Form.Item>
          <Form.Item style={{ marginBottom: 8 }}>
            <Button type="primary" htmlType="submit" block loading={submitting} style={{ height: 44, fontWeight: 600 }}>
              登 录
            </Button>
          </Form.Item>
        </Form>

        <div style={{ textAlign: 'center', marginTop: 12 }}>
          <Text type="secondary" style={{ fontSize: 11 }}>
            单管理员模式 · 凭据由服务端环境变量配置
          </Text>
        </div>
      </div>
    </div>
  )
}
