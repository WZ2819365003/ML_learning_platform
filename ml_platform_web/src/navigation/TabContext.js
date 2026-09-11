import { createContext, useContext, useEffect, useRef } from 'react'

export const TabContext = createContext({ active: true, registerGuard: () => () => {}, markSaved: () => {} })
export function useTabActive() { return useContext(TabContext).active }
export function useTabGuard(guard) {
  const { registerGuard } = useContext(TabContext)
  const ref = useRef(guard)
  ref.current = guard
  useEffect(() => registerGuard(() => typeof ref.current === 'function' ? ref.current() : ref.current), [registerGuard])
}
export function useMarkTabSaved() { return useContext(TabContext).markSaved }
