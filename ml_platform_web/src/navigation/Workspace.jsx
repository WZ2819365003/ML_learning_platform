import { Suspense, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { App, ConfigProvider, Spin } from 'antd'
import { useLocation, useNavigate, useNavigationType } from 'react-router-dom'
import { CloseOutlined, DoubleLeftOutlined, DoubleRightOutlined } from '@ant-design/icons'
import { TabContext } from './TabContext'
import { describeRoute, HOME, persistTabs, restoreTabs, visitTab } from './routes'
import ErrorBoundary from '../components/ErrorBoundary'
import { PageRoutes } from './PageRoutes'

function WorkspacePane({ tab, active, registerPane }) {
  const root = useRef(null)
  const portal = useRef(null)
  const guards = useRef(new Set())
  const registerGuard = useCallback(fn => {
    guards.current.add(fn)
    return () => guards.current.delete(fn)
  }, [])
  const getPortalContainer = useCallback(() => portal.current || document.body, [])
  const markSaved = useCallback(form => guards.current.forEach(fn => {
    const guard = fn()
    if (!form || guard?.form === form) guard?.reset?.()
  }), [])
  useEffect(() => registerPane(tab.id, () => {
    const values = [...guards.current].map(fn => fn() || {})
    return {
      dirty: values.some(g => g.dirty),
      busy: values.some(g => g.busy) || !!root.current?.querySelector('.ant-btn-loading, .ant-upload-list-item-uploading'),
    }
  }), [tab.id, registerPane])
  const context = useMemo(() => ({ active, registerGuard, getPortalContainer, markSaved }), [active, registerGuard, getPortalContainer, markSaved])
  return <section ref={root} className="workspace-pane" role="tabpanel" id={`pane-${tab.id}`} aria-label={tab.title} hidden={!active}>
    <TabContext.Provider value={context}>
      <ConfigProvider getPopupContainer={getPortalContainer} getTargetContainer={getPortalContainer}>
        <div className="workspace-page-scroll">
          <ErrorBoundary scope={tab.title} homeTo={HOME}>
            <Suspense fallback={<div className="page-loading"><Spin size="large" /><span>页面加载中</span></div>}>
              <div className="page-shell"><PageRoutes location={tab.location} /></div>
            </Suspense>
          </ErrorBoundary>
        </div>
        <div ref={portal} className="workspace-portals" />
      </ConfigProvider>
    </TabContext.Provider>
  </section>
}

export default function Workspace() {
  const location = useLocation()
  const navigate = useNavigate()
  const navigationType = useNavigationType()
  const route = describeRoute(location)
  const [tabs, setTabs] = useState(() => visitTab(restoreTabs(window.sessionStorage), route))
  const previousId = useRef(route.id)
  const sessionEnding = useRef(false)
  const panes = useRef(new Map())
  const rail = useRef(null)
  const [scrollable, setScrollable] = useState({ left: false, right: false })
  const { modal, message } = App.useApp()
  // Derive the newly selected pane immediately, avoiding a stale route render
  // between the router update and committing the cache metadata.
  const displayedTabs = visitTab(tabs, route, navigationType, previousId.current)
  useLayoutEffect(() => {
    const previous = previousId.current
    setTabs(current => visitTab(current, describeRoute(location), navigationType, previous))
    previousId.current = describeRoute(location).id
  }, [location, navigationType])
  useEffect(() => { if (!sessionEnding.current) persistTabs(window.sessionStorage, tabs) }, [tabs])
  const updateScroll = useCallback(() => {
    const el = rail.current
    if (!el) return
    const next = { left: el.scrollLeft > 1, right: el.scrollLeft + el.clientWidth < el.scrollWidth - 1 }
    setScrollable(previous => previous.left === next.left && previous.right === next.right ? previous : next)
  }, [])
  const revealActiveTab = useCallback(() => {
    const el = rail.current
    const active = el?.querySelector('.is-active')
    if (active) {
      const view = el.getBoundingClientRect(), item = active.getBoundingClientRect()
      if (item.left < view.left) el.scrollLeft -= view.left - item.left
      else if (item.right > view.right) el.scrollLeft += item.right - view.right
    }
    updateScroll()
  }, [updateScroll])
  useLayoutEffect(revealActiveTab, [route.id, tabs.length, revealActiveTab])
  useEffect(() => {
    const el = rail.current
    const onWheel = event => {
      if (el.scrollWidth <= el.clientWidth) return
      const delta = Math.abs(event.deltaX) > Math.abs(event.deltaY) ? event.deltaX : event.deltaY
      const next = Math.max(0, Math.min(el.scrollWidth - el.clientWidth, el.scrollLeft + delta))
      if (next === el.scrollLeft) return
      event.preventDefault()
      el.scrollLeft = next
    }
    const observer = new ResizeObserver(revealActiveTab)
    observer.observe(el)
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => { observer.disconnect(); el.removeEventListener('wheel', onWheel) }
  }, [revealActiveTab])
  const registerPane = useCallback((id, read) => {
    panes.current.set(id, read)
    return () => panes.current.delete(id)
  }, [])
  useEffect(() => {
    const beforeUnload = event => {
      if (sessionEnding.current) return
      if ([...panes.current.values()].some(read => { const g = read(); return g.dirty || g.busy })) {
        event.preventDefault()
        event.returnValue = ''
      }
    }
    const endSession = () => { sessionEnding.current = true }
    window.addEventListener('beforeunload', beforeUnload)
    window.addEventListener('ml-platform:session-ended', endSession)
    return () => {
      window.removeEventListener('beforeunload', beforeUnload)
      window.removeEventListener('ml-platform:session-ended', endSession)
    }
  }, [])

  const confirmClose = (closing, action) => {
    const status = closing.map(t => panes.current.get(t.id)?.() || {})
    if (status.some(g => g.busy)) {
      message.info('页面正在保存或上传，请完成后再关闭页签')
      return
    }
    if (status.some(g => g.dirty)) {
      modal.confirm({ title: '关闭未保存的页面？', content: '尚未保存的内容将丢失，后台训练任务会继续运行。', okText: '放弃修改并关闭', cancelText: '继续编辑', onOk: action })
    } else action()
  }
  const close = id => {
    const index = displayedTabs.findIndex(t => t.id === id)
    confirmClose([displayedTabs[index]], () => {
      const next = displayedTabs.filter(t => t.id !== id)
      if (route.id === id) {
        const target = next[Math.min(index, next.length - 1)]?.location
        navigate(target || HOME, { state: target?.state })
      }
      setTabs(next)
    })
  }
  const closeAll = () => confirmClose(displayedTabs, () => {
    // A generation key also remounts the default pane when close-all resets it.
    setTabs([{ ...describeRoute({ pathname: HOME, search: '', hash: '' }), visited: true, generation: Date.now() }])
    navigate(HOME)
  })
  const move = delta => rail.current?.scrollBy({ left: delta, behavior: 'smooth' })
  const handleTabKey = (event, index) => {
    const keys = { ArrowLeft: (index + displayedTabs.length - 1) % displayedTabs.length,
      ArrowRight: (index + 1) % displayedTabs.length, Home: 0, End: displayedTabs.length - 1 }
    if (!(event.key in keys)) return
    event.preventDefault()
    const target = displayedTabs[keys[event.key]]
    navigate(target.location, { state: target.location.state })
    rail.current.querySelectorAll('[role="tab"]')[keys[event.key]]?.focus()
  }
  return <main className="workspace">
    <nav className="workspace-tabs" aria-label="已打开的页面">
      <button type="button" className="tabs-arrow" aria-label="向左滚动页签" disabled={!scrollable.left} onClick={() => move(-220)}><DoubleLeftOutlined /></button>
      <div ref={rail} className="workspace-tabs-rail" role="tablist" aria-label="页面页签" onScroll={updateScroll}>
        {displayedTabs.map((tab, index) => <div key={tab.id} className={`workspace-tab ${tab.id === route.id ? 'is-active' : ''}`}>
          <button type="button" role="tab" aria-selected={tab.id === route.id} aria-controls={`pane-${tab.id}`} title={tab.title}
            tabIndex={tab.id === route.id ? 0 : -1} onKeyDown={event => handleTabKey(event, index)}
            onClick={() => navigate(tab.location, { state: tab.location.state })}>{tab.title}</button>
          <button type="button" className="tab-close" title={`关闭${tab.title}`} aria-label={`关闭${tab.title}`} disabled={displayedTabs.length === 1 && tab.id === HOME} onClick={() => close(tab.id)}><CloseOutlined /></button>
        </div>)}
      </div>
      <button type="button" className="tabs-arrow" aria-label="向右滚动页签" disabled={!scrollable.right} onClick={() => move(220)}><DoubleRightOutlined /></button>
      <button type="button" className="tabs-close-all" aria-label="关闭所有页签" onClick={closeAll}>关闭全部</button>
    </nav>
    <div className="workspace-panes">
      {displayedTabs.filter(tab => tab.visited).map(tab => <WorkspacePane key={`${tab.id}:${tab.generation || 0}`} tab={tab} active={tab.id === route.id} registerPane={registerPane} />)}
    </div>
  </main>
}
