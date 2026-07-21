import { describe, expect, it } from 'vitest'
import { environmentTag, failedTaskCount, usernameLabel } from './Header'

describe('Header live status helpers', () => {
  it('shows PROD only for the production environment', () => {
    expect(environmentTag('production')).toEqual({ color: 'blue', label: 'PROD' })
    expect(environmentTag('development')).toEqual({ color: 'default', label: 'DEV' })
    expect(environmentTag()).toEqual({ color: 'default', label: 'DEV' })
  })

  it('normalises failed task totals and usernames', () => {
    expect(failedTaskCount({ total: 7 })).toBe(7)
    expect(failedTaskCount({ total: -1 })).toBe(0)
    expect(failedTaskCount()).toBe(0)
    expect(usernameLabel({ username: 'opsadmin' })).toBe('opsadmin')
    expect(usernameLabel({})).toBe('管理员')
  })
})
