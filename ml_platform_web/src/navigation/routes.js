export const HOME = '/v3/tasks'
export const TABS_STORAGE_KEY = 'ml_platform_workspace_v1'
export const menuGroups = [
  { key: '/data', label: '数据管理', icon: 'data' },
  { key: 'modeling', label: '建模', icon: 'modeling', children: [
    { key: HOME, label: '任务列表' }, { key: '/v3/training-plans', label: '训练方案' },
    { key: '/models', label: '模型管理' }, { key: '/deploy', label: '模型部署' }, { key: '/v3/runs', label: '运行诊断' },
  ] },
  { key: 'ts', label: '时序任务', icon: 'ts', children: [
    { key: '/ts/tasks?drawer=create', label: '新建任务' }, { key: '/ts/tasks', label: '任务列表' },
  ] },
  { key: '/settings', label: '系统设置', icon: 'settings' },
]
const aliases = { '/': HOME, '/dashboard': HOME, '/v3': HOME, '/experiments': HOME, '/tasks': '/v3/runs',
  '/timesfm': '/ts/tasks', '/ts/monitor': '/ts/tasks', '/ts/results': '/ts/tasks',
  '/ts/config': '/ts/tasks?drawer=create', '/ts/tasks/new': '/ts/tasks?drawer=create', '/results': '/training/results' }
const pageTitles = { '/data': '数据管理', [HOME]: '建模任务', '/v3/training-plans': '训练方案',
  '/models': '模型管理', '/deploy': '模型部署', '/v3/runs': '运行诊断', '/settings': '系统设置', '/ts/tasks': '时序任务',
  '/training/config': '训练配置', '/training/monitor': '训练监控', '/training/results': '训练结果',
  '/dl/config': '深度学习配置', '/dl/monitor': '深度学习监控' }

export function canonicalLocation(location) {
  const target = aliases[location.pathname]
  if (target) {
    const [pathname, search] = target.split('?')
    return canonicalLocation({ ...location, pathname, search: search ? `?${search}` : location.search || '' })
  }
  if (location.pathname === '/dl/results') {
    const query = new URLSearchParams(location.search)
    query.set('family', 'dl')
    return canonicalLocation({ ...location, pathname: '/training/results', search: `?${query}` })
  }
  if (location.pathname === '/training/results' && !new URLSearchParams(location.search).get('taskId')) {
    return { ...location, pathname: '/models', search: '' }
  }
  return location
}

export function describeRoute(raw) {
  const location = canonicalLocation(raw)
  const { pathname, search = '' } = location
  const query = new URLSearchParams(search)
  let id = pathname
  let title = pageTitles[pathname]
  let menu = pathname
  const task = pathname.match(/^\/v3\/tasks\/([^/]+)(\/workflow)?$/)
  const ts = pathname.match(/^\/ts\/tasks\/([^/]+)$/)
  if (task) {
    title = task[1] === 'new' ? '新建建模任务' : `${task[2] ? '建模工作流' : '任务详情'} · ${task[1]}`
    menu = HOME
  } else if (ts) {
    title = `时序结果 · ${ts[1]}`
    menu = '/ts/tasks'
  } else if (/^\/(training|dl)\/(monitor|results)$/.test(pathname)) {
    const taskId = query.get('taskId')
    const family = pathname === '/training/results' ? (query.get('family')?.toLowerCase() === 'dl' ? 'dl' : 'ml') : ''
    id += `${family ? `:${family}` : ''}${taskId ? `:${taskId}` : ''}`
    title += taskId ? ` · ${taskId}` : ''
    menu = HOME
  } else if (pathname.startsWith('/experiments/')) {
    title = '实验详情'
    menu = HOME
  } else if (pathname.includes('/config')) menu = HOME
  if (pathname === '/ts/tasks' && query.get('drawer') === 'create') menu = '/ts/tasks?drawer=create'
  return { id, title: title || '页面未找到', menu, location }
}

// Cache identity excludes transient filters, paging and the workflow's step.
export function visitTab(tabs, route, navigationType, previousId) {
  const shouldReplace = navigationType === 'REPLACE' && previousId && (
    previousId === '/v3/tasks/new/workflow' || previousId.startsWith('/experiments/'))
  const existing = tabs.find(t => t.id === route.id)
  if (existing) return tabs.filter(t => !shouldReplace || t.id !== previousId || t.id === route.id)
    .map(t => t.id === route.id ? { ...t, location: route.location, visited: true } : t)
  const tab = { ...route, visited: true }
  return shouldReplace ? tabs.map(t => t.id === previousId ? tab : t) : [...tabs, tab]
}

export function restoreTabs(storage) {
  try {
    const parsed = JSON.parse(storage.getItem(TABS_STORAGE_KEY))
    if (!Array.isArray(parsed)) return []
    const seen = new Set()
    return parsed.slice(0, 30).flatMap(item => {
      if (typeof item?.pathname !== 'string' || !item.pathname.startsWith('/') || item.pathname.startsWith('//') || item.pathname === '/login') return []
      const tab = describeRoute({ pathname: item.pathname, search: typeof item.search === 'string' ? item.search : '', hash: '' })
      if (seen.has(tab.id) || tab.title === '页面未找到') return []
      seen.add(tab.id)
      return [{ ...tab, visited: false }]
    })
  } catch { return [] }
}

export function persistTabs(storage, tabs) {
  try { storage.setItem(TABS_STORAGE_KEY, JSON.stringify(tabs.map(t => ({ pathname: t.location.pathname, search: t.location.search })))) } catch { /* Private mode / quota: navigation still works. */ }
}
