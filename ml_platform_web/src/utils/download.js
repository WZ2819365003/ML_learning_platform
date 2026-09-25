/**
 * 带登录凭证下载文件。
 *
 * 平台所有 /api、/inference 请求都要在请求头里带 Bearer token（后端 main.py 的
 * auth_guard）。以前的下载按钮是 <a href="/api/models/xxx/download">，浏览器直接
 * 打开链接时不会带这个请求头，后端一律 401，用户看到的就是「点了下载没反应 / 下载
 * 下来是一段 JSON」。
 *
 * 这里改成用带凭证的请求把文件取回来，再在浏览器里触发保存。不把 token 拼进 URL
 * 查询参数：那样凭证会进 nginx 访问日志和浏览器历史。
 */

/** 从 Content-Disposition 里取文件名，优先 RFC 5987 的 filename*=。 */
export function filenameFromContentDisposition(header) {
  if (!header) return null
  const star = /filename\*\s*=\s*([^']*)'[^']*'([^;]+)/i.exec(header)
  if (star) {
    try {
      return decodeURIComponent(star[2].trim().replace(/^"|"$/g, ''))
    } catch {
      // 编码坏了就退回普通 filename
    }
  }
  const plain = /filename\s*=\s*("([^"]*)"|[^;]+)/i.exec(header)
  if (plain) return (plain[2] ?? plain[1]).trim() || null
  return null
}

/** 失败时响应体是 Blob（因为请求时要的是 blob），把后端的 detail 读出来。 */
export async function errorMessageFromBlob(data, fallback) {
  if (!data || typeof data.text !== 'function') return fallback
  try {
    const text = await data.text()
    const parsed = JSON.parse(text)
    return typeof parsed?.detail === 'string' ? parsed.detail : fallback
  } catch {
    return fallback
  }
}

/**
 * @param {string} url       站内相对路径，例如 /api/models/{id}/download
 * @param {object} deps      { getToken, onUnauthorized, fetchImpl }，便于测试注入
 * @param {string} fallbackName 响应头没给文件名时用
 */
export async function downloadWithAuth(url, { getToken, onUnauthorized, fetchImpl = fetch } = {}, fallbackName = 'download') {
  const token = getToken?.()
  const response = await fetchImpl(url, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  })

  if (response.status === 401) {
    onUnauthorized?.()
    throw new Error('登录已过期，请重新登录后再下载')
  }
  if (!response.ok) {
    const blob = await response.blob().catch(() => null)
    throw new Error(await errorMessageFromBlob(blob, `下载失败（HTTP ${response.status}）`))
  }

  const blob = await response.blob()
  const name = filenameFromContentDisposition(response.headers.get('content-disposition')) || fallbackName
  saveBlob(blob, name)
  return name
}

/**
 * 把浏览器里已有的 Blob 存成文件。
 *
 * 两个坑：a 元素要挂到 DOM 上再点（部分浏览器对游离节点的 click 不触发下载）；
 * 不能 click 之后立刻 revokeObjectURL——下载是异步开始的，Safari / Firefox 会拿到
 * 一个已经失效的地址，存下来是空文件。
 */
export function saveBlob(blob, name) {
  const objectUrl = URL.createObjectURL(blob)
  try {
    const a = document.createElement('a')
    a.href = objectUrl
    a.download = name
    a.style.display = 'none'
    document.body.appendChild(a)
    a.click()
    a.remove()
  } finally {
    setTimeout(() => URL.revokeObjectURL(objectUrl), 30_000)
  }
}
