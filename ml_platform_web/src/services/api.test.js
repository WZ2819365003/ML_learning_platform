import { afterEach, describe, expect, it, vi } from 'vitest'
import { clearAuthToken } from './api'

afterEach(() => vi.unstubAllGlobals())
describe('Session cleanup', () => {
  it('clears the token and navigation cache, and bypasses stale draft prompts on expiry', () => {
    const localRemove = vi.fn(), sessionRemove = vi.fn(), dispatchEvent = vi.fn()
    vi.stubGlobal('localStorage', { removeItem: localRemove })
    vi.stubGlobal('sessionStorage', { removeItem: sessionRemove })
    vi.stubGlobal('window', { dispatchEvent })
    clearAuthToken()
    expect(localRemove).toHaveBeenCalledWith('ml_platform_token')
    expect(sessionRemove).toHaveBeenCalledWith('ml_platform_workspace_v1')
    expect(dispatchEvent.mock.calls[0][0].type).toBe('ml-platform:session-ended')
  })
})
