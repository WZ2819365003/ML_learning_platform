// Shared API helpers for Playwright probes.

const BASE_API = process.env.BASE_API || 'http://127.0.0.1:8000/api';
const BASE_ROOT = process.env.BASE_ROOT || 'http://127.0.0.1:8000';

async function getJson(request, path) {
  const url = path.startsWith('http') ? path : `${BASE_API}${path}`;
  const res = await request.get(url);
  const body = await res.text();
  let parsed = null;
  try { parsed = body ? JSON.parse(body) : null; } catch { /* leave null */ }
  return { ok: res.ok(), status: res.status(), body: parsed, raw: body };
}

async function postJson(request, path, data) {
  const url = path.startsWith('http') ? path : `${BASE_API}${path}`;
  const res = await request.post(url, { data });
  const body = await res.text();
  let parsed = null;
  try { parsed = body ? JSON.parse(body) : null; } catch { /* leave null */ }
  return { ok: res.ok(), status: res.status(), body: parsed, raw: body };
}

async function listDatasets(request) {
  const r = await getJson(request, '/data/list?page=1&page_size=50');
  if (!r.ok) return [];
  return r.body?.items || r.body?.datasets || [];
}

async function getHealth(request) {
  const res = await request.get(`${BASE_ROOT}/health`);
  return { ok: res.ok(), status: res.status(), body: await res.json().catch(() => null) };
}

module.exports = { BASE_API, BASE_ROOT, getJson, postJson, listDatasets, getHealth };
