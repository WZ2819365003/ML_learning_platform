import React, { useEffect, useState } from 'react'
import { Avatar, Dropdown, Space, Badge, Tag } from '../../ui'
import { BellOutlined, UserOutlined, DownOutlined } from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'
import { authApi, clearAuthToken, platformTasksApi, systemApi } from '../../services/api'

export function environmentTag(environment) {
  return String(environment).toLowerCase() === 'production'
    ? { color: 'blue', label: 'PROD' }
    : { color: 'default', label: 'DEV' }
}

export function failedTaskCount(payload) {
  const total = Number(payload?.total)
  return Number.isFinite(total) && total > 0 ? Math.floor(total) : 0
}

export function usernameLabel(payload) {
  return typeof payload?.username === 'string' && payload.username.trim()
    ? payload.username.trim()
    : '管理员'
}


export default function Header() {
  const navigate = useNavigate()
  const [environment, setEnvironment] = useState('development')
  const [failedCount, setFailedCount] = useState(0)
  const [username, setUsername] = useState('管理员')
  const [userMenuOpen, setUserMenuOpen] = useState(false)
  const envTag = environmentTag(environment)
  useEffect(() => {
    let active = true

    systemApi.health()
      .then(payload => {
        if (active) setEnvironment(payload?.environment || 'development')
      })
      .catch(() => {})
    platformTasksApi.list({ status: 'FAILED', page: 1, page_size: 1 })
      .then(payload => {
        if (active) setFailedCount(failedTaskCount(payload))
      })
      .catch(() => {})
    authApi.me()
      .then(payload => {
        if (active) setUsername(usernameLabel(payload))
      })
      .catch(() => {})

    return () => { active = false }
  }, [])

  const onUserMenuClick = ({ key }) => {
    if (key === 'logout') {
      clearAuthToken()
      window.location.assign('/login')
    }
  }


  return <header className="app-header">
    <div className="app-brand"><span className="brand-symbol" aria-hidden="true">M</span><span>ML Platform<span className="brand-subtitle">智能建模平台</span></span></div>
    <Space size={16} className="header-actions">
      <Tag color={envTag.color}>{envTag.label}</Tag>
      <Badge count={failedCount} offset={[0, 1]}>
        <button className="header-icon" aria-label={failedCount ? `${failedCount} 个失败任务` : '无失败任务'} onClick={() => navigate('/v3/runs')}><BellOutlined /></button>
      </Badge>
      <Dropdown
        menu={{ items: [{ key: 'logout', label: '退出登录' }], onClick: onUserMenuClick }}
        overlayClassName="header-user-menu"
        placement="bottomRight"
        trigger={['click']}
        open={userMenuOpen}
        onOpenChange={setUserMenuOpen}
        autoFocus
      >
        <button className="header-user" aria-label={`${username}，用户菜单`} aria-haspopup="menu" aria-expanded={userMenuOpen}>
          <Avatar size={28} icon={<UserOutlined />} /><span>{username}</span><DownOutlined />
        </button>
      </Dropdown>
    </Space>
  </header>
}
