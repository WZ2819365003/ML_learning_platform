import React from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { Layout } from 'antd'
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

const { Content } = Layout

function App() {
  return (
    <Router>
      <Layout style={{ minHeight: '100vh' }}>
        <Sidebar />
        <Layout>
          <Header />
          <Content style={{ margin: '24px 16px', padding: 24, background: '#fff', minHeight: 280 }}>
            <Routes>
              <Route path="/" element={<Navigate to="/dashboard" />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/data" element={<DataManagement />} />
              <Route path="/training/config" element={<TrainingConfig />} />
              <Route path="/training/monitor" element={<TrainingMonitor />} />
              <Route path="/results" element={<Results />} />
              <Route path="/models" element={<ModelManagement />} />
              <Route path="/deploy" element={<ModelDeploy />} />
              <Route path="/settings" element={<Settings />} />
              <Route path="/dl/config"  element={<DLConfig />} />
              <Route path="/dl/monitor" element={<DLMonitor />} />
              <Route path="/dl/results" element={<DLResults />} />
            </Routes>
          </Content>
        </Layout>
      </Layout>
    </Router>
  )
}

export default App