import React from 'react'
import { Layout, Button, Badge, Avatar, Dropdown, Space, Input } from 'antd'
import { BellOutlined, SearchOutlined, UserOutlined } from '@ant-design/icons'

const { Header: AntHeader } = Layout
const { Search } = Input

const Header = () => {
  const userMenu = [
    {
      key: 'profile',
      label: '个人资料'
    },
    {
      key: 'settings',
      label: '账号设置'
    },
    {
      key: 'logout',
      label: '退出登录'
    }
  ]

  return (
    <AntHeader style={{ background: '#fff', padding: 0, boxShadow: '0 2px 8px rgba(0, 0, 0, 0.09)' }}>
      <div className="flex items-center justify-between h-full px-6">
        <div className="flex items-center">
          <Search
            placeholder="搜索..."
            allowClear
            style={{ width: 300, marginRight: 16 }}
            prefix={<SearchOutlined />}
          />
        </div>
        <div className="flex items-center">
          <Button type="text" icon={<BellOutlined />} className="mr-4">
            <Badge count={5} />
          </Button>
          <Dropdown menu={{ items: userMenu }} placement="bottomRight">
            <Space>
              <Avatar icon={<UserOutlined />} />
              <span className="hidden md:inline">管理员</span>
            </Space>
          </Dropdown>
        </div>
      </div>
    </AntHeader>
  )
}

export default Header