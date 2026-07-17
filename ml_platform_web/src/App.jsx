import { lazy, Suspense } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { App as AntApp, ConfigProvider, Layout, Spin } from 'antd'
import Sidebar from './components/layout/Sidebar'
import Header from './components/layout/Header'
import ErrorBoundary from './components/ErrorBoundary'

const Login = lazy(() => import('./pages/Login'))
const Dashboard = lazy(() => import('./pages/Dashboard'))
const DataManagement = lazy(() => import('./pages/DataManagement'))
const TrainingConfig = lazy(() => import('./pages/TrainingConfig'))
const TrainingMonitor = lazy(() => import('./pages/TrainingMonitor'))
const Results = lazy(() => import('./pages/Results'))
const ModelManagement = lazy(() => import('./pages/ModelManagement'))
const ModelDeploy = lazy(() => import('./pages/ModelDeploy'))
const Settings = lazy(() => import('./pages/Settings'))
const DLConfig = lazy(() => import('./pages/DLConfig'))
const DLMonitor = lazy(() => import('./pages/DLMonitor'))
const DLResults = lazy(() => import('./pages/DLResults'))
const TSConfig = lazy(() => import('./pages/TSConfig'))
const TSMonitor = lazy(() => import('./pages/TSMonitor'))
const TSResults = lazy(() => import('./pages/TSResults'))
const ExperimentRedirect = lazy(() => import('./pages/ExperimentRedirect'))
const ModelingTasks = lazy(() => import('./pages/ModelingTasks'))
const ModelingTaskDetail = lazy(() => import('./pages/ModelingTaskDetail'))
const ModelingWorkflow = lazy(() => import('./pages/ModelingWorkflow'))
const TrainingPlans = lazy(() => import('./pages/TrainingPlans'))
const V3Runs = lazy(() => import('./pages/V3Runs'))

const { Content } = Layout

const antTheme = {
  token: {
    colorPrimary: '#2563eb',
    colorSuccess: '#10b981',
    colorWarning: '#f59e0b',
    colorError: '#ef4444',
    colorInfo: '#3b82f6',
    borderRadius: 8,
    borderRadiusLG: 12,
    fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    fontSize: 13,
    colorBgContainer: '#ffffff',
    colorBgLayout: '#f0f5fb',
    colorBorder: 'rgba(148, 163, 184, 0.3)',
    colorTextHeading: '#0f172a',
    colorTextSecondary: '#64748b',
  },
  components: {
    Layout: { siderBg: 'transparent', headerBg: 'transparent' },
    Menu: {
      darkItemBg: 'transparent',
      darkSubMenuItemBg: 'rgba(255,255,255,0.04)',
      darkItemSelectedBg: 'rgba(59, 130, 246, 0.18)',
      darkItemSelectedColor: '#60a5fa',
      darkItemHoverBg: 'rgba(255,255,255,0.07)',
      darkItemHoverColor: '#ffffff',
      darkItemColor: 'rgba(255,255,255,0.72)',
    },
    Table: { headerBg: '#f8fafc', rowHoverBg: 'rgba(59, 130, 246, 0.04)' },
    Card: { borderRadiusLG: 16 },
  },
}

/* Everything except /login renders inside the app chrome. Access control is
   server-side (401 → axios interceptor redirects to /login), so this shell
   needs no client-side token check of its own. */
function AppShell() {
  return (
    <Layout style={{ minHeight: '100vh', background: '#f0f5fb' }}>
      <Sidebar />
      <Layout style={{ background: 'transparent' }}>
        <Header />
        <Content style={{ margin: '0 20px 20px', padding: 0, background: 'transparent' }}>
          <Suspense fallback={(
            <div style={{ display: 'grid', placeItems: 'center', minHeight: 320 }}>
              <Spin size="large" tip="页面加载中" />
            </div>
          )}>
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/data" element={<DataManagement />} />
              <Route path="/training/config" element={<TrainingConfig />} />
              <Route path="/training/monitor" element={<TrainingMonitor />} />
              <Route path="/results" element={<Navigate to="/training/results" replace />} />
              <Route path="/training/results" element={<Results />} />
              <Route path="/models" element={<ModelManagement />} />
              <Route path="/deploy" element={<ModelDeploy />} />
              <Route path="/settings" element={<Settings />} />
              <Route path="/dl/config" element={<DLConfig />} />
              <Route path="/dl/monitor" element={<DLMonitor />} />
              <Route path="/dl/results" element={<DLResults />} />

              <Route path="/timesfm" element={<Navigate to="/ts/tasks" replace />} />
              <Route path="/ts/tasks" element={<TSMonitor />} />
              <Route path="/ts/tasks/new" element={<TSConfig />} />
              <Route path="/ts/tasks/:taskId" element={<TSResults />} />

              <Route path="/ts/config" element={<Navigate to="/ts/tasks/new" replace />} />
              <Route path="/ts/monitor" element={<TSMonitor />} />
              <Route path="/ts/results" element={<TSResults />} />

              {/* Legacy V3 pages retired — redirect into the 建模 group.
                  (/experiments/:id keeps working via ExperimentRedirect below.) */}
              <Route path="/tasks" element={<Navigate to="/v3/runs" replace />} />
              <Route path="/experiments" element={<Navigate to="/v3/tasks" replace />} />
              <Route path="/experiments/:experimentId" element={<ExperimentRedirect />} />

              {/* V3 Modeling Workbench — new task-centric workflow */}
              <Route path="/v3/tasks" element={<ModelingTasks />} />
              {/* Guided linear workflow (数据→配置→训练→可视化→部署). taskId="new" = create mode. */}
              <Route path="/v3/tasks/new/workflow" element={
                <ErrorBoundary scope="建模工作流" homeTo="/v3/tasks">
                  <ModelingWorkflow />
                </ErrorBoundary>
              } />
              <Route path="/v3/tasks/:taskId/workflow" element={
                <ErrorBoundary scope="建模工作流" homeTo="/v3/tasks">
                  <ModelingWorkflow />
                </ErrorBoundary>
              } />
              {/* ErrorBoundary: detail page has many sub-tabs that can throw
                  during data fetch; isolate so one tab crash doesn't white-screen. */}
              <Route path="/v3/tasks/:taskId" element={
                <ErrorBoundary scope="建模任务详情" homeTo="/v3/tasks">
                  <ModelingTaskDetail />
                </ErrorBoundary>
              } />
            <Route path="/v3/training-plans" element={<TrainingPlans />} />
            <Route path="/v3/runs" element={<V3Runs />} />
            <Route path="/v3" element={<Navigate to="/v3/tasks" replace />} />
          </Routes>
          </Suspense>
        </Content>
      </Layout>
    </Layout>
  )
}

function App() {
  return (
    <ConfigProvider theme={antTheme}>
    <AntApp>
    <Router>
      <Suspense fallback={null}>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/*" element={<AppShell />} />
        </Routes>
      </Suspense>
    </Router>
    </AntApp>
    </ConfigProvider>
  )
}

export default App
