import { useEffect } from 'react'
import { useTabActive } from '../navigation/TabContext'

// Same dependency contract as useEffect, with cleanup whenever the cached page
// is hidden. Resuming reruns the caller's refresh/subscription setup.
export function useActiveEffect(effect, dependencies) {
  const active = useTabActive()
  // Callers supply their effect's dependencies, just as they do with useEffect.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => active ? effect() : undefined, [active, ...dependencies])
}
