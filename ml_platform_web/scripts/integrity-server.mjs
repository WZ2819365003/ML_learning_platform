/** Isolated browser QA backend. Only listens on loopback; never proxies writes.
 * Run with Node 20+: node scripts/integrity-server.mjs (UI http://127.0.0.1:3300).
 * Its in-memory records reset on restart. Production imports none of this code. */
import http from 'node:http'
import { spawn } from 'node:child_process'
import { WebSocketServer } from 'ws'
import { overviewReport } from '../src/components/workbench/__fixtures__/reportCharts.fixtures.js'

const now = '2026-09-09T08:00:00Z'
const dataset = { id: 'ds-1', name: '电力负荷样本.csv', row_count: 1200, column_count: 3, file_size: 42000, created_at: now,
  columns_info: { temperature: { dtype: 'float64' }, hour: { dtype: 'int64' }, load: { dtype: 'float64' } } }
let tasks = Array.from({ length: 22 }, (_, i) => ({ id: `task-${i + 1}`, name: `负荷预测任务 ${i + 1}`, description: '日内用电负荷预测',
  task_type: 'regression', dataset_id: dataset.id, dataset_name: dataset.name, target_column: 'load', objective_metric: 'rmse', objective_direction: 'minimize',
  status: i === 1 ? 'RUNNING' : 'COMPLETED', experiment_count: 2, successful_run_count: 3, created_at: now, updated_at: now, experiments: [] }))
const runs = [1, 2].map(i => ({ id: `run-${i}`, run_id: `run-${i}`, task_id: `task-${i}`, task_name: `负荷预测任务 ${i}`, experiment_id: 'exp-1',
  experiment_name: '基线实验', status: i === 1 ? 'SUCCESS' : 'FAILED', strategy_type: 'baseline', model_type: 'random_forest_regressor', task_type: 'regression',
  domain_task_id: `training-${i}`, metrics: { rmse: 0.12, mae: 0.08, r2: 0.93 }, rank: i, started_at: now, finished_at: now, duration_seconds: 24 }))
const tsTask = { id: 'ts-1', name: '日内负荷预测', status: 'SUCCESS', dataset_id: dataset.id, dataset_name: dataset.name, value_column: 'load', horizon: 24, frequency: 'high', created_at: now,
  result: { historical: [110, 120, 115], point_forecast: [125, 132, 127], q10: [120, 127, 122], q90: [130, 137, 132] }, metrics: { rmse: 2.4, mae: 1.2 } }
const list = items => ({ items, total: items.length, datasets: items, deployments: items, models: items, tasks: items, entries: [] })
const training = { id: 'training-1', name: '回归训练样本', status: 'SUCCESS', progress: 100, task_type: 'regression', model_type: 'random_forest_regressor',
  dataset_id: dataset.id, dataset, target_column: 'load', created_at: now, metrics: { rmse: 0.12, mae: 0.08, r2: 0.93 }, family: 'ml', duration: 24 }
let plans = Array.from({ length: 100 }, (_, i) => ({ id: `plan-${i + 1}`, name: `负荷建模方案 ${i + 1}`, description: '组件回归测试样本',
  task_type: i % 2 ? 'classification' : 'regression', model_family: ['ml', 'dl', 'mixed'][i % 3],
  strategy_type: ['baseline', 'grid_search', 'bayesian_search'][i % 3], selected_models: ['random_forest_regressor'],
  default_objective_metric: i % 2 ? 'accuracy' : 'rmse', default_objective_direction: i % 2 ? 'max' : 'min',
  eval_metrics: [i % 2 ? 'accuracy' : 'rmse'], use_count: i, created_at: now }))
let state = { error: '', empty: false, delay: 0, unauthorized: false }
const requests = []
const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, 'http://127.0.0.1:3901'), p = url.pathname.replace(/\/$/, ''), method = req.method
  const reply = (data, status = 200) => { res.writeHead(status, { 'Content-Type': typeof data === 'string' ? 'text/plain; charset=utf-8' : 'application/json' }); res.end(typeof data === 'string' ? data : JSON.stringify(data)) }
  let body = ''; for await (const chunk of req) body += chunk
  let payload = {}; try { payload = JSON.parse(body || '{}') } catch { /* Multipart upload fixture. */ }
  if (p === '/__test/state') { if (method === 'POST') state = { ...state, ...payload }; return reply(state) }
  if (p === '/__test/requests') return reply(requests)
  requests.push({ method, path: p, query: url.search, time: Date.now(), ...(method !== 'GET' ? { payload } : {}) })
  if (state.delay) await new Promise(resolve => setTimeout(resolve, state.delay))
  if (state.unauthorized && p !== '/api/auth/login') return reply({ detail: '登录已过期' }, 401)
  if (state.error && p.includes(state.error)) return reply({ detail: '完整性测试：接口暂时不可用' }, 500)
  if (p === '/health') return reply({ status: 'ok', environment: 'development' })
  if (p === '/api/auth/login') { state.unauthorized = false; return reply({ token: 'local-fixture-only' }) }
  if (p === '/api/auth/me') return reply({ username: '完整性测试' })
  if (p === '/api/v3/tasks') {
    if (method === 'POST') { const task = { ...tasks[0], ...payload, id: `task-${tasks.length + 1}`, status: 'CREATED' }; tasks.unshift(task); return reply(task) }
    const page = Number(url.searchParams.get('page') || 1), size = Number(url.searchParams.get('page_size') || 10)
    return reply({ items: state.empty ? [] : tasks.slice((page - 1) * size, page * size), total: state.empty ? 0 : tasks.length })
  }
  if (p === '/api/v3/runs') return reply(list(state.empty ? [] : runs))
  if (/\/api\/v3\/tasks\/[^/]+$/.test(p)) {
    const id = p.split('/').pop(), item = tasks.find(t => t.id === id)
    if (method === 'DELETE') { tasks = tasks.filter(t => t.id !== id); return reply({ ok: true }) }
    if (method === 'PATCH' && item) Object.assign(item, payload)
    return item ? reply(item) : reply({ detail: '任务不存在' }, 404)
  }
  if (p.endsWith('/leaderboard')) return reply(state.empty ? [] : runs)
  if (p.endsWith('/runs')) return reply(list(state.empty ? [] : runs))
  if (p.endsWith('/progress-tree')) return reply({ task: tasks[0], experiments: [], runs: [], summary: {} })
  if (p.endsWith('/strategy-comparison')) return reply({ strategies: [] })
  if (p.endsWith('/ai-reports')) return reply(list([{ id: 'report-1', created_at: now, summary: '负荷预测评估报告' }]))
  if (p.endsWith('/ai-reports/report-1') || p.endsWith('/ai-report')) return reply(overviewReport)
  if (p.endsWith('/report.md')) return reply('# 负荷预测报告\n\n训练样本的测试报告。')
  if (p.startsWith('/api/v3/tasks/tuning-spaces')) return reply({ models: { random_forest_regressor: { display_name: '随机森林', fixed: {}, grid_values: {}, distribution: {} } } })
  if (p === '/api/data/list') return reply(list(state.empty ? [] : [dataset]))
  if (p.endsWith('/preview')) {
    const rows = Array.from({ length: 100 }, (_, i) => ({ temperature: 26 + i / 10, hour: i % 24, load: 120 + i }))
    return reply({ ...dataset, columns: ['temperature','hour','load'], rows, statistics: {}, data: rows })
  }
  if (p.endsWith('/versions')) return reply(list([]))
  if (p === '/api/data/upload') return reply({ ...dataset, name: '上传验证.csv' })
  if (p === '/api/platform/training-plans') {
    if (method === 'POST') { const plan = { ...payload, id: `plan-${plans.length + 1}`, created_at: now }; plans.push(plan); return reply(plan) }
    return reply(list(state.empty ? [] : plans.filter(plan => !url.searchParams.has('task_type') || plan.task_type === url.searchParams.get('task_type'))))
  }
  if (p === '/api/training/models' || p === '/api/dl/models') return reply({
    models: p === '/api/dl/models'
      ? [{ id: 'mlp', name: '多层感知机', display_name: '多层感知机', task_types: ['classification', 'regression'], arch_params: [] }]
      : [{ id: 'random_forest_regressor', name: '随机森林回归', display_name: '随机森林回归', category: 'ensemble', task_types: ['regression'], hyperparameters: [] }],
    categories: [{ id: 'ensemble', name: '集成学习' }], classification_metrics: [{ label: '准确率', value: 'accuracy' }], regression_metrics: [{ label: 'RMSE', value: 'rmse' }], optimizer_params: [], train_params: [] })
  if (/\/api\/models\/[^/]+\/detail$/.test(p)) return reply({ ...training, result_metrics: training.metrics })
  if (p.endsWith('/trained-models') || p === '/api/models/assets') return reply(list([]))
  if (p === '/api/models/tags') return reply({ tags: [], dimensions: [] })
  if (p === '/api/ts/tasks' || p === '/api/timesfm/list' || (p === '/api/platform/tasks' && url.searchParams.get('orphan_only'))) {
    if (method === 'POST') return reply({ ...tsTask, ...payload })
    const count = state.empty ? 0 : 32, page = Number(url.searchParams.get('page') || 1), size = Number(url.searchParams.get('page_size') || 20)
    return reply({ items: Array.from({length: count}, (_, i) => ({ ...tsTask, id: `ts-${i + 1}`, kind: 'predict', name: i === 0 ? tsTask.name : `预测任务 ${i + 1}` })).slice((page - 1) * size, page * size), total: count })
  }
  if (p === '/api/training/start' || p === '/api/dl/train') return reply({ ...training, ...payload })
  if (p.startsWith('/api/platform/experiments/')) return reply({ id: 'exp-1', modeling_task_id: 'task-1' })
  if (p === '/api/ts/tasks/ts-1') return reply(tsTask)
  if (p.endsWith('/model/status')) return reply({ loaded: false, available: true, status: 'not_loaded', backends: [] })
  if (p.endsWith('/deployments') || p.startsWith('/api/deploy/')) return reply(list([]))
  if (p === '/api/training/list' || p === '/api/dl/list') return reply(list([training]))
  if (/\/api\/(training|dl)\/[^/]+\/status$/.test(p)) return reply({ ...training, id: p.split('/')[3], family: p.includes('/dl/') ? 'dl' : 'ml' })
  if (p.endsWith('/epochs') || p.endsWith('/logs') || p.startsWith('/api/logs/')) return reply({ entries: [], items: [], total: 0 })
  if (p.endsWith('/inspector')) return reply({ run: runs[0], modeling_task: { ...tasks[0], rank: 1 }, experiment: { id: 'exp-1', name: '基线实验' },
    training_task: training, logs: [{ level: 'INFO', message: '训练完成，模型已保存', created_at: now }], siblings: [], diagnosis: {}, shap: { status: 'not_available' } })
  if (p.endsWith('/shap')) return reply({ status: 'success', method: 'shap', feature_names: ['temperature', 'hour'],
    feature_importances: { temperature: 0.8, hour: 0.2 }, feature_count: 2, sample_size: 3,
    ...(state.shapSamples ? { samples: { feature_values: [[26, 1], [27, 2], [28, 3]], shap_values: [[0.7, 0.1], [0.8, 0.2], [0.9, -0.1]] } } : {}) })
  if (p.startsWith('/api/platform/tasks')) return reply(list([]))
  return reply(list([]))
})
const sockets = new WebSocketServer({ server })
sockets.on('connection', (socket, request) => {
  requests.push({ method: 'WS_OPEN', path: request.url.split('?')[0], time: Date.now() })
  socket.send(JSON.stringify({ type: 'log', level: 'INFO', message: '实时日志连接成功', timestamp: now }))
  socket.on('close', () => requests.push({ method: 'WS_CLOSE', path: request.url.split('?')[0], time: Date.now() }))
})
server.listen(3901, '127.0.0.1', () => console.log('Isolated fixture API: http://127.0.0.1:3901'))
const vite = spawn(process.execPath, ['node_modules/vite/bin/vite.js', '--host', '127.0.0.1', '--port', '3300', '--strictPort'], { stdio: 'inherit', env: { ...process.env, VITE_API_TARGET: 'http://127.0.0.1:3901' } })
const stop = () => { vite.kill('SIGTERM'); server.close(); process.exit(0) }
process.on('SIGINT', stop); process.on('SIGTERM', stop)
vite.on('exit', stop)
