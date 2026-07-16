import React, { useState } from 'react'
import { Grid, Layout, Menu, Tooltip } from 'antd'
import { Link, useLocation } from 'react-router-dom'
import {
  DashboardOutlined,
  DatabaseOutlined,
  RocketOutlined,
  SettingOutlined,
  LineChartOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
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
  const screens = Grid.useBreakpoint()
  const isMobile = screens.lg === false

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
      key: 'modeling',
      icon: <RocketOutlined />,
      label: '建模',
      children: [
        { key: 'modeling-tasks', label: <Link to="/v3/tasks">任务列表</Link> },
        { key: 'models',         label: <Link to="/models">模型管理</Link> },
        { key: 'deploy',         label: <Link to="/deploy">模型部署</Link> },
        { key: 'v3-runs',        label: <Link to="/v3/runs">运行诊断</Link> },
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
      type: 'divider',
    },
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
    // /v3/tasks(/…/workflow) is the primary 建模 → 任务列表 entry
    if (path === '/v3/tasks' || path.startsWith('/v3/tasks/')) return 'modeling-tasks'
    if (path === '/v3/runs' || path.startsWith('/v3/runs/')) return 'v3-runs'
    if (path === '/dl/config') return 'dl-config'
    if (path === '/dl/monitor') return 'dl-monitor'
    if (path === '/dl/results') return 'dl-results'
    if (path === '/ts/tasks/new' || path === '/ts/config') return 'ts-new'
    if (path.startsWith('/ts/tasks') || path === '/ts/monitor' || path === '/ts/results') return 'ts-list'
    return 'dashboard'
  }

  return <>
    {isMobile && !collapsed && (
      <div
        aria-hidden="true"
        onClick={() => setCollapsed(true)}
        style={{
          position: 'fixed',
          inset: 0,
          background: 'rgba(15, 23, 42, 0.35)',
          zIndex: 99,
        }}
      />
    )}
    <Sider
      collapsible
      collapsed={collapsed}
      onCollapse={setCollapsed}
      breakpoint="lg"
      onBreakpoint={setCollapsed}
      trigger={null}
      width={220}
      collapsedWidth={72}
      style={{
        background: 'linear-gradient(180deg, #090f1e 0%, #0d1b38 60%, #0f2046 100%)',
        boxShadow: '4px 0 24px rgba(9, 15, 30, 0.45)',
        borderRight: 'none',
        position: isMobile ? 'fixed' : 'relative',
        inset: isMobile ? '0 auto 0 0' : undefined,
        height: isMobile ? '100dvh' : undefined,
        zIndex: isMobile ? 100 : 10,
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
        defaultOpenKeys={['modeling', 'ts']}
        items={menuItems}
        onClick={() => { if (isMobile) setCollapsed(true) }}
        style={{
          background: 'transparent',
          borderRight: 0,
          flex: 1,
          overflow: 'hidden auto',
        }}
      />

      {/* Collapse toggle */}
      <button
        type="button"
        aria-label={collapsed ? '展开侧栏' : '收起侧栏'}
        onClick={() => setCollapsed(!collapsed)}
        style={{
          position: 'absolute',
          bottom: 20,
          left: 0,
          right: 0,
          display: 'flex',
          justifyContent: 'center',
          cursor: 'pointer',
          padding: 0,
          border: 0,
          background: 'transparent',
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
      </button>
    </Sider>
    {isMobile && <div aria-hidden="true" style={{ width: 72, flex: '0 0 72px' }} />}
  </>
}

export default Sidebar
