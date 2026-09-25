/* eslint-disable react-refresh/only-export-components -- Compatibility barrel intentionally re-exports the Ant Design API. */
// Ant Design behavior stays intact. Only portal ownership and draft protection
// are added here so cached pages cannot leak dialogs into another workspace tab.
import { createContext, forwardRef, useContext, useRef } from 'react'
import { Form as AntForm, Modal as AntModal, Drawer as AntDrawer } from 'antd'
import { TabContext, useTabGuard } from '../navigation/TabContext'
export * from 'antd'

const OverlayContext = createContext(true)

export const Form = Object.assign(forwardRef(function WorkspaceForm({ onValuesChange, ...props }, ref) {
  const dirty = useRef(false)
  const visible = useContext(OverlayContext)
  useTabGuard(() => ({
    form: props.form,
    dirty: visible && dirty.current && (!props.form || props.form.isFieldsTouched()),
    reset: () => { dirty.current = false },
  }))
  return <AntForm {...props} ref={ref} onValuesChange={(...args) => {
    dirty.current = true
    onValuesChange?.(...args)
  }} />
}), { Item: AntForm.Item, List: AntForm.List, ErrorList: AntForm.ErrorList, Provider: AntForm.Provider,
  useForm: AntForm.useForm, useFormInstance: AntForm.useFormInstance, useWatch: AntForm.useWatch })

export const Modal = Object.assign(function WorkspaceModal({ children, ...props }) {
  const { getPortalContainer } = useContext(TabContext)
  const dirty = useRef(false)
  if (!props.open) dirty.current = false
  useTabGuard(() => ({ dirty: props.open && dirty.current, busy: props.open && props.confirmLoading }))
  return <AntModal getContainer={getPortalContainer} modalRender={node => <div onChangeCapture={() => { dirty.current = true }}>{node}</div>} {...props}>
    <OverlayContext.Provider value={!!props.open}>{children}</OverlayContext.Provider>
  </AntModal>
}, AntModal)

export const Drawer = Object.assign(function WorkspaceDrawer({ children, ...props }) {
  const { getPortalContainer } = useContext(TabContext)
  return <AntDrawer getContainer={getPortalContainer} {...props}>
    <OverlayContext.Provider value={!!props.open}>{children}</OverlayContext.Provider>
  </AntDrawer>
}, AntDrawer)
