import React from 'react'
import { Layout, Avatar, Dropdown, Space, Badge, Typography, Tag } from 'antd'
import { BellOutlined, UserOutlined, QuestionCircleOutlined, HomeOutlined } from '@ant-design/icons'
import { useLocation } from 'react-router-dom'
import { clearAuthToken } from '../../services/api'

const { Header: AntHeader } = Layout
const { Text } = Typography

// Leaf page title per route. Section (first breadcrumb crumb) is derived from
// the path prefix in getBreadcrumb() so the header stays in sync with Sidebar.
const PAGE_TITLES = {
  '/dashboard':        '仪表盘',
  '/data':             '数据管理',
  '/training/config':  '训练配置',
  '/training/monitor': '训练监控',
  '/training/results': '结果可视化',
  '/models':           '模型管理',
  '/deploy':           '模型部署',
  '/settings':         '系统设置',
  '/dl/config':        '模型配置',
  '/dl/monitor':       '训练监控',
  '/dl/results':       '结果可视化',
  '/ts/tasks/new':     '新建任务',
  '/ts/tasks':         '任务列表',
  '/v3/tasks':         '建模工作台',
  '/v3/training-plans':'训练方案',
  '/v3/runs':          '运行诊断',
}

// Path prefix -> section label (matches the Sidebar groups). Order matters:
// more specific prefixes first. All /v3/* now belong to the 建模 group.
const SECTIONS = [
  ['/v3',       '建模'],
  ['/models',   '建模'],
  ['/deploy',   '建模'],
  ['/training', '建模'],
  ['/dl',       '建模'],
  ['/ts',       '时序任务'],
]

// Returns an array of crumb strings, e.g. ['机器学习', '训练配置'].
function getBreadcrumb(pathname) {
  let leaf = PAGE_TITLES[pathname]
  if (!leaf) {
    if (pathname.endsWith('/workflow')) leaf = '建模工作流'
    else if (pathname.startsWith('/ts/tasks/')) leaf = '任务详情'
    else if (pathname.startsWith('/experiments/')) leaf = '实验详情'
    else if (pathname.startsWith('/v3/tasks/')) leaf = '任务详情'
    else leaf = 'ML Platform'
  }
  const section = SECTIONS.find(([p]) => pathname.startsWith(p))?.[1]
  return section && section !== leaf ? [section, leaf] : [leaf]
}

const userMenu = [
  { key: 'profile', label: '个人资料' },
  { key: 'settings', label: '账号设置' },
  { type: 'divider' },
  { key: 'logout', label: '退出登录', danger: true },
]

const Header = () => {
  const location = useLocation()
  const crumbs = getBreadcrumb(location.pathname)

  const onUserMenuClick = ({ key }) => {
    if (key === 'logout') {
      clearAuthToken()
      window.location.assign('/login')
    }
  }

  return (
    <AntHeader
      style={{
        background: 'rgba(240, 245, 251, 0.85)',
        backdropFilter: 'blur(12px)',
        WebkitBackdropFilter: 'blur(12px)',
        borderBottom: '1px solid rgba(148, 163, 184, 0.15)',
        padding: '0 24px',
        height: 56,
        lineHeight: 'normal',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        position: 'sticky',
        top: 0,
        zIndex: 9,
      }}
    >
      {/* Left: breadcrumb — navigational context, complements the page's own H1 */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0 }}>
        <HomeOutlined style={{ color: '#94a3b8', fontSize: 14 }} />
        {crumbs.map((c, i) => (
          <React.Fragment key={i}>
            {i > 0 && <span style={{ color: '#cbd5e1', fontSize: 13 }}>/</span>}
            <Text
              style={{
                fontSize: 13,
                fontWeight: i === crumbs.length - 1 ? 600 : 400,
                color: i === crumbs.length - 1 ? '#334155' : '#94a3b8',
                whiteSpace: 'nowrap',
              }}
            >
              {c}
            </Text>
          </React.Fragment>
        ))}
      </div>

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
        <Dropdown menu={{ items: userMenu, onClick: onUserMenuClick }} placement="bottomRight" trigger={['click']}>
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
