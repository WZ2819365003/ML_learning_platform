import { lazy, Suspense, useEffect } from 'react'
import { BrowserRouter, Routes, Route, useLocation, useNavigate } from 'react-router-dom'
import { App as AntApp, ConfigProvider } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import Header from './components/layout/Header'
import Sidebar from './components/layout/Sidebar'
import Workspace from './navigation/Workspace'
import { canonicalLocation } from './navigation/routes'
import { darkTheme } from './theme/antd'

const Login = lazy(() => import('./pages/Login'))
// Static message/notification/confirm calls in existing business pages inherit
// the same theme and locale as context-based Ant Design controls.
ConfigProvider.config({ holderRender: children => <ConfigProvider theme={darkTheme} locale={zhCN} button={{ autoInsertSpace: false }}><AntApp>{children}</AntApp></ConfigProvider> })

function CanonicalAddress() {
  const location = useLocation()
  const navigate = useNavigate()
  useEffect(() => {
    const canonical = canonicalLocation(location)
    if (canonical.pathname !== location.pathname || canonical.search !== location.search) {
      navigate(canonical, { replace: true, state: location.state })
    }
  }, [location, navigate])
  return null
}

export default function App() {
  return <ConfigProvider theme={darkTheme} locale={zhCN} button={{ autoInsertSpace: false }}>
    <AntApp><BrowserRouter><CanonicalAddress /><Suspense fallback={null}><Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/*" element={<div className="app-shell"><Header /><div className="app-shell-body"><Sidebar /><Workspace /></div></div>} />
    </Routes></Suspense></BrowserRouter></AntApp>
  </ConfigProvider>
}
