import React from 'react'
import { Result, Button, Typography } from 'antd'
import { ReloadOutlined, HomeOutlined } from '@ant-design/icons'
import { Link } from 'react-router-dom'

const { Paragraph, Text } = Typography

/**
 * Generic React error boundary.
 *
 * Wrap any page / sub-tree whose crash shouldn't white-screen the whole app.
 * Typical use: `<ErrorBoundary><SomeLazyPage/></ErrorBoundary>`.  Rendering
 * errors bubble up to componentDidCatch; network errors thrown inside
 * useEffect don't — those still need their own try/catch.
 *
 * Props:
 *   - `scope`   : human-readable label shown in the fallback UI
 *   - `homeTo`  : where the 「回首页」 button navigates (default /dashboard)
 */
export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { error: null, info: null }
  }

  static getDerivedStateFromError(error) {
    return { error }
  }

  componentDidCatch(error, info) {
    // Log to console; in prod we could wire this to an error tracker.
    console.error('[ErrorBoundary]', this.props.scope || 'unscoped', error, info)
    this.setState({ info })
  }

  handleReload = () => {
    // Full page reload — simpler than resetting component state for most
    // "my page is broken" scenarios.
    window.location.reload()
  }

  handleReset = () => {
    this.setState({ error: null, info: null })
  }

  render() {
    const { error, info } = this.state
    const { children, scope, homeTo = '/dashboard' } = this.props

    if (!error) return children

    const stack = info?.componentStack || error?.stack || ''

    return (
      <div style={{ padding: 24 }}>
        <Result
          status="error"
          title={scope ? `${scope} 页面渲染错误` : '页面渲染错误'}
          subTitle="页面中的某个组件抛出了异常；这不会影响其他页面。可以尝试刷新或回到首页。"
          extra={[
            <Button type="primary" icon={<ReloadOutlined />} key="reload" onClick={this.handleReload}>
              刷新页面
            </Button>,
            <Button key="reset" onClick={this.handleReset}>
              重试此页
            </Button>,
            <Link to={homeTo} key="home">
              <Button icon={<HomeOutlined />}>回首页</Button>
            </Link>,
          ]}
        >
          <div style={{ background: '#f8fafc', padding: 16, borderRadius: 8, marginTop: 12 }}>
            <Paragraph>
              <Text strong style={{ fontSize: 13 }}>错误信息：</Text>
              <Text code copyable style={{ fontSize: 12, wordBreak: 'break-all' }}>
                {error?.message || String(error)}
              </Text>
            </Paragraph>
            {stack && (
              <details>
                <summary style={{ cursor: 'pointer', color: '#64748b', fontSize: 12 }}>
                  查看调用栈
                </summary>
                <pre style={{
                  marginTop: 8, fontSize: 11, padding: 10, borderRadius: 6,
                  background: '#0f172a', color: '#e2e8f0', overflow: 'auto',
                  maxHeight: 320,
                }}>
                  {stack}
                </pre>
              </details>
            )}
          </div>
        </Result>
      </div>
    )
  }
}
