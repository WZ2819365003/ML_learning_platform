import React, { useState } from 'react'
import { Layout, Menu, Avatar } from 'antd'
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
} from '@ant-design/icons'

const { Sider } = Layout

const Sidebar = () => {
  const location = useLocation()
  const [collapsed, setCollapsed] = useState(false)

  const menuItems = [
    {
      key: 'dashboard',
      icon: <DashboardOutlined />,
      label: <Link to="/dashboard">仪表盘</Link>
    },
    {
      key: 'data',
      icon: <DatabaseOutlined />,
      label: <Link to="/data">数据管理</Link>
    },
    {
      key: 'training',
      icon: <RocketOutlined />,
      label: '机器学习',
      children: [
        { key: 'training-config', label: <Link to="/training/config">训练配置</Link> },
        { key: 'training-monitor', label: <Link to="/training/monitor">训练监控</Link> },
        { key: 'training-results', label: <Link to="/training/results">结果可视化</Link> }
      ]
    },
    {
      key: 'dl',
      icon: <ExperimentOutlined />,
      label: '深度学习',
      children: [
        { key: 'dl-config', label: <Link to="/dl/config">模型配置</Link> },
        { key: 'dl-monitor', label: <Link to="/dl/monitor">训练监控</Link> },
        { key: 'dl-results', label: <Link to="/dl/results">结果可视化</Link> }
      ]
    },
    {
      key: 'ts',
      icon: <LineChartOutlined />,
      label: '时序预测',
      children: [
        { key: 'ts-config',  label: <Link to="/ts/config">预测配置</Link> },
        { key: 'ts-monitor', label: <Link to="/ts/monitor">任务监控</Link> },
        { key: 'ts-results', label: <Link to="/ts/results">结果可视化</Link> }
      ]
    },
    {
      key: 'models',
      icon: <AppstoreOutlined />,
      label: <Link to="/models">模型管理</Link>
    },
    {
      key: 'deploy',
      icon: <CloudServerOutlined />,
      label: <Link to="/deploy">模型部署</Link>
    },
    {
      key: 'settings',
      icon: <SettingOutlined />,
      label: <Link to="/settings">系统设置</Link>
    }
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
    if (path === '/dl/config') return 'dl-config'
    if (path === '/dl/monitor') return 'dl-monitor'
    if (path === '/dl/results') return 'dl-results'
    if (path === '/ts/config')  return 'ts-config'
    if (path === '/ts/monitor') return 'ts-monitor'
    if (path === '/ts/results') return 'ts-results'
    return 'dashboard'
  }

  return (
    <Sider
      collapsible
      collapsed={collapsed}
      onCollapse={setCollapsed}
      style={{
        background: '#fff',
        boxShadow: '2px 0 8px rgba(0, 0, 0, 0.09)'
      }}
    >
      <div className="flex flex-col items-center pt-6 pb-4">
        <Avatar size={48} style={{ backgroundColor: '#1890ff' }}>ML</Avatar>
        {!collapsed && <h2 className="mt-2 text-xl font-semibold">ML Platform</h2>}
      </div>
      <Menu
        mode="inline"
        selectedKeys={[getSelectedKey()]}
        defaultOpenKeys={['training', 'dl', 'ts']}
        items={menuItems}
        style={{ borderRight: 0 }}
      />
    </Sider>
  )
}

export default Sidebar
