/**
 * 把后端返回的网关相对路径拼成可以直接复制调用的绝对地址。
 *
 * 后端不知道自己被哪个域名/端口访问，所以推理入口一律返回相对路径
 * （/api/ts/deployments/{id}/predict、/inference/{id}/predict）。之前时序那条
 * 写死了 http://127.0.0.1:8000，页面上显示的是容器自己的回环地址，谁都调不通。
 *
 * 已经是绝对地址的（老数据、外部网关）原样返回。
 */
export function absoluteEndpoint(path, origin) {
  if (!path) return ''
  const value = String(path)
  if (/^[a-z][a-z0-9+.-]*:\/\//i.test(value)) return value

  const base = origin
    ?? (typeof window === 'undefined' ? '' : window.location?.origin)
    ?? ''
  if (!base) return value
  return `${base.replace(/\/+$/, '')}/${value.replace(/^\/+/, '')}`
}

export default absoluteEndpoint
