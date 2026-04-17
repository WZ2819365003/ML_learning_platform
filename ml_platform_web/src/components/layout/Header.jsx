import React from 'react'
import { Layout, Avatar, Dropdown, Space, Badge, Typography, Tag } from 'antd'
import { BellOutlined, UserOutlined, QuestionCircleOutlined } from '@ant-design/icons'
import { useLocation } from 'react-router-dom'

const { Header: AntHeader } = Layout
const { Text } = Typography

const PAGE_TITLES = {
  '/dashboard':        { title: '仪表盘', sub: '概览所有训练任务与模型指标' },
  '/data':             { title: '数据管理', sub: '上传、预览和管理训练数据集' },
  '/training/config':  { title: '训练配置', sub: '配置并启动机器学习训练任务' },
  '/training/monitor': { title: '训练监控', sub: '实时跟踪训练进度与日志' },
  '/training/results': { title: '结果可视化', sub: '分析和对比模型性能指标' },
  '/models':           { title: '模型管理', sub: '查看和管理已保存的模型' },
  '/deploy':           { title: '模型部署', sub: '将模型发布为推理接口' },
  '/settings':         { title: '系统设置', sub: '配置平台参数和环境变量' },
  '/dl/config':        { title: '深度学习配置', sub: '配置神经网络结构与超参数' },
  '/dl/monitor':       { title: '深度学习监控', sub: '逐 Epoch 跟踪损失与指标' },
  '/dl/results':       { title: '深度学习结果', sub: '分析训练曲线与模型表现' },
  '/ts/tasks/new':     { title: '新建时序任务', sub: '使用 Chronos 进行时间序列预测' },
  '/ts/tasks':         { title: '时序任务列表', sub: '管理所有时间序列预测任务' },
  '/tasks':            { title: '任务中心', sub: '统一管理所有异步任务 — 训练、解释、评估' },
  '/experiments':      { title: '实验管理', sub: '组织多次训练 Run，对比模型性能，自动筛选最优' },
}

function getPageMeta(pathname) {
  if (PAGE_TITLES[pathname]) return PAGE_TITLES[pathname]
  if (pathname.startsWith('/ts/tasks/')) return { title: '时序任务详情', sub: '查看预测结果与统计信息' }
  if (pathname.startsWith('/experiments/')) return { title: '实验详情', sub: '排行榜 · 多模型指标对比' }
  return { title: 'ML Platform', sub: '' }
}

const userMenu = [
  { key: 'profile', label: '个人资料' },
  { key: 'settings', label: '账号设置' },
  { type: 'divider' },
  { key: 'logout', label: '退出登录', danger: true },
]

const Header = () => {
  const location = useLocation()
  const { title, sub } = getPageMeta(location.pathname)

  return (
    <AntHeader
      style={{
        background: 'rgba(240, 245, 251, 0.85)',
        backdropFilter: 'blur(12px)',
        WebkitBackdropFilter: 'blur(12px)',
        borderBottom: '1px solid rgba(148, 163, 184, 0.15)',
        padding: '0 24px',
        height: 64,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        position: 'sticky',
        top: 0,
        zIndex: 9,
      }}
    >
      {/* Left: page title */}
      <Space direction="vertical" size={0}>
        <Text strong style={{ fontSize: 16, color: '#0f172a', lineHeight: 1.25 }}>{title}</Text>
        {sub && <Text style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.3 }}>{sub}</Text>}
      </Space>

      {/* Right: actions */}
      <Space size={4} align="center">
        {/* Env indicator */}
        <Tag color="blue" style={{ marginRight: 8, borderRadius: 6, fontSize: 11, fontWeight: 600 }}>
          PROD
        </Tag>

        {/* Notifications */}
        <Badge dot offset={[-4, 4]} color="#3b82f6">
          <div
            style={{
              width: 36,
              height: 36,
              borderRadius: 8,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#64748b',
              cursor: 'pointer',
              transition: 'background 0.2s',
              fontSize: 16,
            }}
            onMouseEnter={e => e.currentTarget.style.background = 'rgba(59,130,246,0.08)'}
            onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
          >
            <BellOutlined />
          </div>
        </Badge>

        {/* Help */}
        <div
          style={{
            width: 36,
            height: 36,
            borderRadius: 8,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#64748b',
            cursor: 'pointer',
            transition: 'background 0.2s',
            fontSize: 16,
          }}
          onMouseEnter={e => e.currentTarget.style.background = 'rgba(59,130,246,0.08)'}
          onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
        >
          <QuestionCircleOutlined />
        </div>

        {/* User avatar */}
        <Dropdown menu={{ items: userMenu }} placement="bottomRight" trigger={['click']}>
          <Space
            style={{ cursor: 'pointer', padding: '4px 8px', borderRadius: 8, transition: 'background 0.2s' }}
            onMouseEnter={e => e.currentTarget.style.background = 'rgba(59,130,246,0.08)'}
            onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
          >
            <Avatar
              size={30}
              icon={<UserOutlined />}
              style={{
                background: 'linear-gradient(135deg, #2563eb, #3b82f6)',
                fontSize: 13,
              }}
            />
            <Text style={{ fontSize: 13, color: '#0f172a', fontWeight: 500 }} className="hidden md:inline">
              管理员
            </Text>
          </Space>
        </Dropdown>
      </Space>
    </AntHeader>
  )
}

export default Header
