import React from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { App as AntApp, Layout } from 'antd'
import Sidebar from './components/layout/Sidebar'
import Header from './components/layout/Header'
import Dashboard from './pages/Dashboard'
import DataManagement from './pages/DataManagement'
import TrainingConfig from './pages/TrainingConfig'
import TrainingMonitor from './pages/TrainingMonitor'
import Results from './pages/Results'
import ModelManagement from './pages/ModelManagement'
import ModelDeploy from './pages/ModelDeploy'
import Settings from './pages/Settings'
import DLConfig from './pages/DLConfig'
import DLMonitor from './pages/DLMonitor'
import DLResults from './pages/DLResults'
import TSConfig from './pages/TSConfig'
import TSMonitor from './pages/TSMonitor'
import TSResults from './pages/TSResults'

const { Content } = Layout

function App() {
  return (
    <AntApp>
    <Router>
      <Layout style={{ minHeight: '100vh' }}>
        <Sidebar />
        <Layout>
          <Header />
          <Content style={{ margin: '24px 16px', padding: 24, background: '#fff', minHeight: 280 }}>
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
            </Routes>
          </Content>
        </Layout>
      </Layout>
    </Router>
    </AntApp>
  )
}

export default App
