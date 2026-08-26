import { describe, expect, it } from 'vitest'

import { resolveBackTarget } from './ModelDetail'

/** Stand-in for URLSearchParams-like access. */
const search = (obj) => ({ get: (k) => (k in obj ? obj[k] : null) })

// The whole point of the standalone page is that it is reached from two
// places that must return to different ones. Getting this wrong strands the
// user on a page whose back button lies.
describe('resolveBackTarget', () => {
  it('returns to the workflow training step when opened from the workflow', () => {
    const t = resolveBackTarget({ state: { from: 'workflow', taskId: 'abc' }, search: search({}) })
    expect(t.label).toBe('返回训练过程')
    // step=2 is 训练过程; the workflow reads it from the query string.
    expect(t.to).toBe('/v3/tasks/abc/workflow?step=2')
  })

  it('returns to model management when opened from there', () => {
    const t = resolveBackTarget({ state: { from: 'models' }, search: search({}) })
    expect(t.label).toBe('返回模型管理')
    expect(t.to).toBe('/models')
  })

  it('falls back to the query string when router state was lost', () => {
    // A refresh or a pasted deep link drops router state entirely.
    const t = resolveBackTarget({ state: null, search: search({ from: 'workflow', taskId: 'xyz' }) })
    expect(t.to).toBe('/v3/tasks/xyz/workflow?step=2')
  })

  it('never dead-ends when there is no origin at all', () => {
    const t = resolveBackTarget({ state: null, search: search({}) })
    expect(t.to).toBe('/models')
  })

  it('degrades to the task list when the workflow origin has no task id', () => {
    const t = resolveBackTarget({ state: { from: 'workflow' }, search: search({}) })
    expect(t.to).toBe('/v3/tasks')
  })
})
