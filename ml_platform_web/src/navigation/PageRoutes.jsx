import { lazy } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { Result, Button } from '../ui'
import ErrorBoundary from '../components/ErrorBoundary'
const DataManagement = lazy(() => import('../pages/DataManagement'))
const TrainingConfig = lazy(() => import('../pages/TrainingConfig'))
const TrainingMonitor = lazy(() => import('../pages/TrainingMonitor'))
const UnifiedResults = lazy(() => import('../pages/UnifiedResults'))
const ModelManagement = lazy(() => import('../pages/ModelManagement'))
const ModelDeploy = lazy(() => import('../pages/ModelDeploy'))
const Settings = lazy(() => import('../pages/Settings'))
const DLConfig = lazy(() => import('../pages/DLConfig'))
const DLMonitor = lazy(() => import('../pages/DLMonitor'))
const TSConfig = lazy(() => import('../pages/TSConfig'))
const TSMonitor = lazy(() => import('../pages/TSMonitor'))
const TSResults = lazy(() => import('../pages/TSResults'))
const ExperimentRedirect = lazy(() => import('../pages/ExperimentRedirect'))
const ModelingTasks = lazy(() => import('../pages/ModelingTasks'))
const ModelingTaskDetail = lazy(() => import('../pages/ModelingTaskDetail'))
const ModelingWorkflow = lazy(() => import('../pages/ModelingWorkflow'))
const TrainingPlans = lazy(() => import('../pages/TrainingPlans'))
const V3Runs = lazy(() => import('../pages/V3Runs'))


export function PageRoutes({ location }) {
 return (
          <Routes location={location}>
            {/* Landing page is 建模 → 任务列表. The old 仪表盘 page read the retired
                legacy TrainingTask tables and was removed; its path is kept only
                as a redirect so old bookmarks still land somewhere useful. */}
            <Route path="/" element={<Navigate to="/v3/tasks" replace />} />
              <Route path="/dashboard" element={<Navigate to="/v3/tasks" replace />} />
              <Route path="/data" element={<DataManagement />} />
              <Route path="/training/config" element={<TrainingConfig />} />
              <Route path="/training/monitor" element={<TrainingMonitor />} />
              <Route path="/results" element={<Navigate to="/training/results" replace />} />
              <Route path="/training/results" element={<UnifiedResults />} />
              <Route path="/models" element={<ModelManagement />} />
              <Route path="/deploy" element={<ModelDeploy />} />
              <Route path="/settings" element={<Settings />} />
              <Route path="/dl/config" element={<DLConfig />} />
              <Route path="/dl/monitor" element={<DLMonitor />} />
              <Route path="/dl/results" element={<UnifiedResults />} />

              <Route path="/timesfm" element={<Navigate to="/ts/tasks" replace />} />
              <Route path="/ts/tasks" element={<TSMonitor />} />
              <Route path="/ts/tasks/new" element={<TSConfig />} />
              <Route path="/ts/tasks/:taskId" element={<TSResults />} />

              <Route path="/ts/config" element={<Navigate to="/ts/tasks/new" replace />} />
              <Route path="/ts/monitor" element={<TSMonitor />} />
              {/* Retired unscoped result entry: a result always belongs to a task. */}
              <Route path="/ts/results" element={<Navigate to="/ts/tasks" replace />} />

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
          <Route path="*" element={<Result status="404" title="页面未找到" extra={<Button href="/v3/tasks">返回建模任务</Button>} />} /></Routes>
 )
}
