import { describe, expect, it } from 'vitest'

import { _activeNodeKeys, isActive } from './ProgressTree'

// `isActive` is the single predicate behind three UI decisions that must agree
// with the backend: 停止 is only offered while a Run can still be cancelled,
// 删除 is blocked while a batch is RUNNING (the server refuses it outright, see
// experiment_service.delete_experiment), and the log modal only claims to be
// tailing live when the run really is.
describe('isActive', () => {
  it('treats every non-terminal scheduler status as active', () => {
    // These are the statuses task_runner can still advance out of.
    for (const status of ['RUNNING', 'PENDING', 'QUEUED', 'RETRY']) {
      expect(isActive(status), status).toBe(true)
    }
  })

  it('treats terminal statuses as inactive', () => {
    // SUCCESS/COMPLETED/FAILED/CANCELLED are terminal — offering 停止 here
    // would call cancel_task, which rejects an already-terminal task with a
    // 400 the user cannot act on.
    for (const status of ['SUCCESS', 'COMPLETED', 'FAILED', 'CANCELLED']) {
      expect(isActive(status), status).toBe(false)
    }
  })

  it('is case-insensitive', () => {
    // The progress tree surfaces ExperimentRun.status verbatim; ML and DL
    // paths have not always agreed on casing.
    expect(isActive('running')).toBe(true)
    expect(isActive('Running')).toBe(true)
  })

  it('treats a missing status as inactive rather than throwing', () => {
    // A run that has not been dispatched yet can arrive with a null status;
    // rendering must not crash the whole tree over it.
    expect(isActive(undefined)).toBe(false)
    expect(isActive(null)).toBe(false)
    expect(isActive('')).toBe(false)
  })
})

// The tree used to be pinned fully expanded with no way to collapse, so every
// "再加一组" permanently added a batch row and its runs. Auto-expansion now
// covers only what the scheduler can still advance, which is what keeps the
// panel a fixed height as batches accumulate.
describe('_activeNodeKeys', () => {
  const tree = (experiments) => ({ experiments })

  it('expands a batch that is itself still active', () => {
    const keys = _activeNodeKeys(tree([{ id: 'e1', status: 'RUNNING', runs: [] }]))
    expect(keys).toEqual(['exp:e1'])
  })

  it('expands a finished batch that still has a live run', () => {
    // The batch row can settle before its last trial does; hiding the run that
    // is still going would be exactly the wrong thing to collapse.
    const keys = _activeNodeKeys(tree([
      { id: 'e1', status: 'COMPLETED', runs: [{ status: 'RUNNING' }] },
    ]))
    expect(keys).toEqual(['exp:e1'])
  })

  it('leaves fully finished batches collapsed', () => {
    const keys = _activeNodeKeys(tree([
      { id: 'e1', status: 'COMPLETED', runs: [{ status: 'SUCCESS' }] },
      { id: 'e2', status: 'COMPLETED', runs: [{ status: 'FAILED' }] },
    ]))
    expect(keys).toEqual([])
  })

  it('keeps the panel bounded as batches accumulate', () => {
    // Five finished batches and one running: only the running one opens.
    const experiments = [1, 2, 3, 4, 5].map((n) => ({
      id: `done${n}`, status: 'COMPLETED', runs: [{ status: 'SUCCESS' }],
    }))
    experiments.push({ id: 'live', status: 'RUNNING', runs: [{ status: 'RUNNING' }] })
    expect(_activeNodeKeys(tree(experiments))).toEqual(['exp:live'])
  })

  it('tolerates a tree with no experiments yet', () => {
    expect(_activeNodeKeys(tree(undefined))).toEqual([])
    expect(_activeNodeKeys(null)).toEqual([])
  })
})
