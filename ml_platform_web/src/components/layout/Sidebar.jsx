import React, { useState } from 'react'
import { Layout, Menu, Tooltip } from 'antd'
import { Link, useLocation } from 'react-router-dom'
import {
  DashboardOutlined,
  DatabaseOutlined,
  RocketOutlined,
  AppstoreOutlined,
  CloudServerOutlined,
  SettingOutlined,
  ExperimentOutlined,
  LineChartOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  ControlOutlined,
  BranchesOutlined,
} from '@ant-design/icons'

const { Sider } = Layout

/* Brand logo mark — rendered as SVG so no external assets needed */
function BrandMark({ size = 36 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="36" height="36" rx="10" fill="url(#grad)" />
      <path d="M10 26 L18 10 L26 26" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" fill="none" opacity="0.9"/>
      <circle cx="18" cy="10" r="2.5" fill="#60a5fa" />
      <circle cx="10" cy="26" r="2" fill="white" opacity="0.7" />
      <circle cx="26" cy="26" r="2" fill="white" opacity="0.7" />
      <defs>
        <linearGradient id="grad" x1="0" y1="0" x2="36" y2="36" gradientUnits="userSpaceOnUse">
          <stop stopColor="#1d4ed8" />
          <stop offset="1" stopColor="#3b82f6" />
        </linearGradient>
      </defs>
    </svg>
  )
}

const Sidebar = () => {
  const location = useLocation()
  const [collapsed, setCollapsed] = useState(false)

  const menuItems = [
    {
      key: 'dashboard',
      icon: <DashboardOutlined />,
      label: <Link to="/dashboard">仪表盘</Link>,
    },
    {
      key: 'data',
      icon: <DatabaseOutlined />,
      label: <Link to="/data">数据管理</Link>,
    },
    {
      key: 'training',
      icon: <RocketOutlined />,
      label: '机器学习',
      children: [
        { key: 'training-config',   label: <Link to="/training/config">训练配置</Link> },
        { key: 'training-monitor',  label: <Link to="/training/monitor">训练监控</Link> },
        { key: 'training-results',  label: <Link to="/training/results">结果可视化</Link> },
      ],
    },
    {
      key: 'dl',
      icon: <ExperimentOutlined />,
      label: '深度学习',
      children: [
        { key: 'dl-config',   label: <Link to="/dl/config">模型配置</Link> },
        { key: 'dl-monitor',  label: <Link to="/dl/monitor">训练监控</Link> },
        { key: 'dl-results',  label: <Link to="/dl/results">结果可视化</Link> },
      ],
    },
    {
      key: 'ts',
      icon: <LineChartOutlined />,
      label: '时序任务',
      children: [
        { key: 'ts-new',  label: <Link to="/ts/tasks/new">新建任务</Link> },
        { key: 'ts-list', label: <Link to="/ts/tasks">任务列表</Link> },
      ],
    },
    {
      key: 'models',
      icon: <AppstoreOutlined />,
      label: <Link to="/models">模型管理</Link>,
    },
    {
      key: 'deploy',
      icon: <CloudServerOutlined />,
      label: <Link to="/deploy">模型部署</Link>,
    },
    {
      type: 'divider',
    },
    // V3 Platform features — hidden behind VITE_ENABLE_V3 flag until Stage 2
    // is fully deployed. Set VITE_ENABLE_V3=true in .env.local to enable.
    ...(import.meta.env.VITE_ENABLE_V3 !== 'false' ? [{
      key: 'v3',
      icon: <BranchesOutlined />,
      label: (
        <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          V3 平台
          <span style={{
            fontSize: 9, fontWeight: 700, letterSpacing: '0.04em',
            background: 'rgba(96,165,250,0.18)', color: '#60a5fa',
            padding: '1px 5px', borderRadius: 4, lineHeight: 1.6,
          }}>
            BETA
          </span>
        </span>
      ),
      children: [
        { key: 'v3-tasks',    label: <Link to="/v3/tasks">建模工作台</Link> },
        { key: 'tasks',       label: <Link to="/tasks">任务中心</Link> },
        { key: 'experiments', label: <Link to="/experiments">实验管理</Link> },
      ],
    }] : []),
    {
      key: 'settings',
      icon: <SettingOutlined />,
      label: <Link to="/settings">系统设置</Link>,
    },
  ]

  const getSelectedKey = () => {
    const path = location.pathname
    if (path === '/dashboard') return 'dashboard'
    if (path === '/data') return 'data'
    if (path === '/training/config') return 'training-config'
    if (path === '/training/monitor') return 'training-monitor'
    if (path === '/training/results') return 'training-results'
    if (path === '/models') return 'models'
    if (path === '/deploy') return 'deploy'
    if (path === '/settings') return 'settings'
    if (path === '/tasks') return 'tasks'
    if (path === '/v3/tasks' || path.startsWith('/v3/tasks/')) return 'v3-tasks'
    if (path === '/experiments' || path.startsWith('/experiments/')) return 'experiments'
    if (path === '/dl/config') return 'dl-config'
    if (path === '/dl/monitor') return 'dl-monitor'
    if (path === '/dl/results') return 'dl-results'
    if (path === '/ts/tasks/new' || path === '/ts/config') return 'ts-new'
    if (path.startsWith('/ts/tasks') || path === '/ts/monitor' || path === '/ts/results') return 'ts-list'
    return 'dashboard'
  }

  return (
    <Sider
      collapsible
      collapsed={collapsed}
      onCollapse={setCollapsed}
      trigger={null}
      width={220}
      collapsedWidth={72}
      style={{
        background: 'linear-gradient(180deg, #090f1e 0%, #0d1b38 60%, #0f2046 100%)',
        boxShadow: '4px 0 24px rgba(9, 15, 30, 0.45)',
        borderRight: 'none',
        position: 'relative',
        zIndex: 10,
      }}
    >
      {/* Logo area */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          padding: collapsed ? '20px 18px' : '20px 20px',
          borderBottom: '1px solid rgba(255,255,255,0.06)',
          marginBottom: 8,
          cursor: 'pointer',
          transition: 'padding 0.2s',
        }}
      >
        <BrandMark size={36} />
        {!collapsed && (
          <div style={{ overflow: 'hidden' }}>
            <div style={{ color: '#fff', fontWeight: 700, fontSize: 15, lineHeight: 1.2, letterSpacing: '-0.02em', whiteSpace: 'nowrap' }}>
              ML Platform
            </div>
            <div style={{ color: 'rgba(255,255,255,0.38)', fontSize: 10, fontWeight: 500, letterSpacing: '0.1em', textTransform: 'uppercase', marginTop: 2 }}>
              Enterprise
            </div>
          </div>
        )}
      </div>

      {/* Nav menu */}
      <Menu
        theme="dark"
        mode="inline"
        selectedKeys={[getSelectedKey()]}
        defaultOpenKeys={['training', 'dl', 'ts', 'v3']}
        items={menuItems}
        style={{
          background: 'transparent',
          borderRight: 0,
          flex: 1,
          overflow: 'hidden auto',
        }}
      />

      {/* Collapse toggle */}
      <div
        onClick={() => setCollapsed(!collapsed)}
        style={{
          position: 'absolute',
          bottom: 20,
          left: 0,
          right: 0,
          display: 'flex',
          justifyContent: 'center',
          cursor: 'pointer',
        }}
      >
        <Tooltip title={collapsed ? '展开' : '收起'} placement="right">
          <div
            style={{
              width: 32,
              height: 32,
              borderRadius: 8,
              background: 'rgba(255,255,255,0.07)',
              border: '1px solid rgba(255,255,255,0.1)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'rgba(255,255,255,0.55)',
              fontSize: 14,
              transition: 'background 0.2s',
            }}
          >
            {collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
          </div>
        </Tooltip>
      </div>
    </Sider>
  )
}

export default Sidebar
