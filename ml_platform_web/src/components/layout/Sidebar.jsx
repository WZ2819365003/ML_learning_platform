import React, { useState } from 'react'
import { Grid, Menu } from '../../ui'
import { useLocation, useNavigate } from 'react-router-dom'
import { DatabaseOutlined, RocketOutlined, SettingOutlined, LineChartOutlined, MenuFoldOutlined, MenuUnfoldOutlined } from '@ant-design/icons'
import { describeRoute, menuGroups } from '../../navigation/routes'

const icons = { data: <DatabaseOutlined />, modeling: <RocketOutlined />, ts: <LineChartOutlined />, settings: <SettingOutlined /> }
const items = menuGroups.map(group => ({ ...group, icon: icons[group.icon] }))

export default function Sidebar() {
  const route = describeRoute(useLocation())
  const navigate = useNavigate()
  const mobile = Grid.useBreakpoint().lg === false
  const [collapsed, setCollapsed] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)
  const [openKeys, setOpenKeys] = useState(['modeling', 'ts'])
  const isCollapsed = mobile ? !mobileOpen : collapsed
  return <>
    {mobile && mobileOpen && <button className="sidebar-mask" aria-label="收起菜单" onClick={() => setMobileOpen(false)} />}
    <aside className={`app-sidebar ${isCollapsed ? 'is-collapsed' : ''} ${mobile ? 'is-mobile' : ''}`} aria-label="主菜单">
      <Menu theme="dark" mode="inline" inlineCollapsed={isCollapsed} inlineIndent={24}
        selectedKeys={[route.menu]} openKeys={isCollapsed ? [] : openKeys} onOpenChange={setOpenKeys} items={items}
        onClick={({ key }) => { navigate(key); if (mobile) setMobileOpen(false) }} />
      {!mobile && <button className="sidebar-toggle" aria-label={collapsed ? '展开侧栏' : '收起侧栏'} onClick={() => setCollapsed(!collapsed)}>
        {collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
      </button>}
    </aside>
    {mobile && <button className="mobile-menu-toggle" aria-label={mobileOpen ? '收起侧栏' : '展开侧栏'} onClick={() => setMobileOpen(!mobileOpen)}><MenuUnfoldOutlined /></button>}
  </>
}
