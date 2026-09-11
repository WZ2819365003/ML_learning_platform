import { describe, expect, it } from 'vitest'
import { canonicalLocation, describeRoute, HOME, menuGroups, persistTabs, restoreTabs, TABS_STORAGE_KEY, visitTab } from './routes'

const route = address => {
  const url = new URL(address, 'https://example.test')
  return describeRoute({ pathname: url.pathname, search: url.search, hash: url.hash })
}
describe('Workspace identity and navigation compatibility', () => {
  it('keeps list filters and workflow steps within one tab', () => {
    expect(route('/v3/runs?status=FAILED').id).toBe(route('/v3/runs?status=SUCCESS').id)
    expect(route('/v3/tasks/task-a/workflow?step=1').id).toBe(route('/v3/tasks/task-a/workflow?step=3').id)
    expect(route('/ts/tasks/new').id).toBe(route('/ts/tasks').id)
  })
  it('isolates task IDs and ML/DL results, including query-based legacy routes', () => {
    for (const base of ['/training/monitor', '/dl/monitor', '/training/results']) {
      expect(route(`${base}?taskId=a`).id).not.toBe(route(`${base}?taskId=b`).id)
    }
    expect(route('/training/results?taskId=a&family=dl').id).not.toBe(route('/training/results?taskId=a').id)
    expect(route('/v3/tasks/a').id).not.toBe(route('/v3/tasks/b').id)
    expect(route('/ts/tasks/a').id).not.toBe(route('/ts/tasks/b').id)
  })
  it('normalizes result family exactly like the existing result renderer', () => {
    expect(route('/training/results?taskId=a&family=DL').id).toBe(route('/dl/results?taskId=a').id)
    expect(route('/training/results?taskId=a&family=invalid').id).toBe(route('/training/results?taskId=a').id)
  })
  it.each([['/', HOME], ['/dashboard', HOME], ['/tasks', '/v3/runs'], ['/experiments', HOME], ['/ts/config', '/ts/tasks'], ['/ts/monitor', '/ts/tasks'], ['/results', '/models'], ['/dl/results', '/models']])('canonicalizes %s without a ghost redirect tab', (path, expected) => {
    expect(route(path).location.pathname).toBe(expected)
    const location = canonicalLocation({ pathname: path, search: '?a=1', state: { preferredDatasetId: 'ds' } })
    expect(location.state).toEqual({ preferredDatasetId: 'ds' })
  })
  it('replaces a new task with the created task, including its next workflow step', () => {
    const initial = [route(HOME), route('/v3/tasks/new/workflow')]
    const result = visitTab(initial, route('/v3/tasks/a/workflow?step=1'), 'REPLACE', '/v3/tasks/new/workflow')
    expect(result.map(t => t.id)).toEqual([HOME, '/v3/tasks/a/workflow'])
    expect(result[1].location.search).toBe('?step=1')
  })
  it('removes a redirect tab even if its destination is already cached', () => {
    const initial = [route('/v3/tasks/a'), route('/experiments/old')]
    expect(visitTab(initial, route('/v3/tasks/a'), 'REPLACE', '/experiments/old').map(t => t.id)).toEqual(['/v3/tasks/a'])
  })
  it('updates navigation state without losing another cached pane', () => {
    const initial = [{ ...route(HOME), visited: true }, route('/v3/runs')]
    const result = visitTab(initial, route('/v3/runs?status=FAILED'))
    expect(result[0]).toBe(initial[0])
    expect(result).toHaveLength(2)
    expect(result[1].location.search).toBe('?status=FAILED')
  })
  it('exposes every primary business menu, including training plans', () => {
    const keys = menuGroups.flatMap(group => group.children?.map(c => c.key) || [group.key])
    expect(keys).toEqual(expect.arrayContaining([HOME, '/v3/training-plans', '/data', '/models', '/deploy', '/v3/runs', '/ts/tasks', '/settings']))
  })
})

describe('Workspace recovery', () => {
  const memory = () => {
    const values = new Map()
    return { getItem: k => values.get(k), setItem: (k, v) => values.set(k, v) }
  }
  it('persists navigation metadata only; restored panes remain unmounted', () => {
    const storage = memory()
    const tab = route('/v3/tasks/a/workflow?step=2')
    tab.location.state = { secretDraft: 'never persist' }
    persistTabs(storage, [tab, route('/v3/runs')])
    expect(storage.getItem(TABS_STORAGE_KEY)).not.toContain('secretDraft')
    const restored = restoreTabs(storage)
    expect(restored).toHaveLength(2)
    expect(restored.every(t => t.visited === false)).toBe(true)
  })
  it('recovers from corrupt/blocked storage and rejects unknown/external entries', () => {
    expect(restoreTabs({ getItem: () => '{broken' })).toEqual([])
    expect(restoreTabs({ getItem: () => { throw Error('blocked') } })).toEqual([])
    expect(() => persistTabs({ setItem: () => { throw Error('quota') } }, [])).not.toThrow()
    const storage = memory()
    storage.setItem(TABS_STORAGE_KEY, JSON.stringify([{ pathname: '//evil.test' }, { pathname: '/login' }, { pathname: '/missing' }, { pathname: HOME }, { pathname: HOME }]))
    expect(restoreTabs(storage).map(t => t.id)).toEqual([HOME])
  })
})
