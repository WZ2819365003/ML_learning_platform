import { describe, expect, it } from 'vitest'

import { isActive } from './ProgressTree'

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
