const WEB_PORT = process.env.E2E_WEB_PORT || '3100';
const API_PORT = process.env.E2E_API_PORT || '8100';

const WEB_BASE = (process.env.BASE_UI || `http://127.0.0.1:${WEB_PORT}`)
  .replace(/\/$/, '');
const API_ROOT = (process.env.BASE_API || `http://127.0.0.1:${API_PORT}`)
  .replace(/\/api\/?$/, '')
  .replace(/\/$/, '');
const API_BASE = `${API_ROOT}/api`;

module.exports = { WEB_BASE, API_ROOT, API_BASE };
