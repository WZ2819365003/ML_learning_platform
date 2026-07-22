# E2E Regression Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Chromium E2E suite run against its own FastAPI/Vite servers and SQLite database, then repair every remaining reproducible test or product failure.

**Architecture:** Centralize E2E URL construction in one helper so every spec interprets `BASE_API`, `E2E_API_PORT`, `BASE_UI`, and `E2E_WEB_PORT` identically. Serialize SQLite-backed runs to prevent concurrent writers from invalidating test results. After infrastructure is deterministic, use each remaining failing spec as the RED case and apply the smallest selector, fixture, or product fix needed.

**Tech Stack:** Playwright Test, Node.js CommonJS, FastAPI, Vite, SQLite/aiosqlite.

## Global Constraints

- Do not reuse the user's Docker services on ports 8000/3000 when override ports are supplied.
- Keep production behavior unchanged unless an isolated E2E failure proves a production defect.
- Every fix must be verified first by its failing spec and finally by the complete Chromium project.
- Preserve the existing Chinese UI copy unless the application, rather than the test, is demonstrably wrong.

---

### Task 1: Centralize E2E server addresses

**Files:**
- Create: `tests/helpers/e2e-env.js`
- Modify: all `tests/*.spec.js` files that define or hardcode `BASE`, `WEB_BASE`, `API`, `API_BASE`, or `BASE_API`
- Modify: `tests/helpers/training-tasks.js`
- Test: the existing Chromium E2E specs

**Interfaces:**
- Produces: `WEB_BASE` (frontend origin), `API_ROOT` (backend origin), and `API_BASE` (backend origin plus `/api`).
- Consumes: `BASE_UI`, `BASE_API`, `E2E_WEB_PORT`, and `E2E_API_PORT` environment variables.

- [x] **Step 1: Record the RED evidence**

Run the complete suite on ports 8099/3099 without `BASE_UI` or `BASE_API`. Confirm error contexts contain requests to ports 8000/3000 and Docker-only paths such as `/app/storage/uploads`.

- [x] **Step 2: Add the shared helper**

```js
const WEB_PORT = process.env.E2E_WEB_PORT || '3000';
const API_PORT = process.env.E2E_API_PORT || '8000';
const WEB_BASE = (process.env.BASE_UI || `http://127.0.0.1:${WEB_PORT}`).replace(/\/$/, '');
const API_ROOT = (process.env.BASE_API || `http://127.0.0.1:${API_PORT}`)
  .replace(/\/api\/?$/, '')
  .replace(/\/$/, '');
const API_BASE = `${API_ROOT}/api`;

module.exports = { WEB_BASE, API_ROOT, API_BASE };
```

- [x] **Step 3: Replace per-file URL definitions**

Import only the needed constants with `require('./helpers/e2e-env')` or `require('./e2e-env')`. Replace hardcoded `http://127.0.0.1:8000` and `http://127.0.0.1:3000` references with `API_ROOT`, `API_BASE`, or `WEB_BASE` according to whether the path already begins with `/api`.

- [x] **Step 4: Verify affected specs use override ports**

Run:

```bash
E2E_API_PORT=8099 E2E_WEB_PORT=3099 E2E_DATABASE_URL='sqlite+aiosqlite:////tmp/ml-platform-e2e-url/e2e.db' npx playwright test --project=chromium --workers=1
```

Expected: no request or navigation in failure output targets ports 8000/3000.

### Task 2: Serialize SQLite-backed E2E execution

**Files:**
- Modify: `playwright.config.js`
- Test: `tests/m3-report-batch-automl.spec.js` plus the complete Chromium project

**Interfaces:**
- Consumes: `E2E_DATABASE_URL`.
- Produces: deterministic worker count for SQLite while retaining configurable parallelism for non-SQLite databases.

- [x] **Step 1: Preserve the RED evidence**

Use the prior full-suite output showing `sqlite3.OperationalError: database is locked` under five workers.

- [x] **Step 2: Configure worker count from the database backend**

```js
const usesSqlite = e2eDatabaseUrl.startsWith('sqlite');
const configuredWorkers = process.env.E2E_WORKERS
  ? Number(process.env.E2E_WORKERS)
  : (usesSqlite ? 1 : undefined);
```

Set Playwright's `workers` to `process.env.CI ? 1 : configuredWorkers` and reject non-positive or non-integer `E2E_WORKERS` values during config loading.

- [x] **Step 3: Verify the AutoML chain without a CLI worker override**

Run the M3 spec with a fresh SQLite URL and no `--workers` option. Expected: output says `using 1 worker` and all four tests pass.

### Task 3: Repair remaining deterministic failures

**Files:**
- Modify only the spec, fixture helper, or production component named by each fresh failure stack.
- Test: each failing spec, then the complete Chromium project.

**Interfaces:**
- Consumes: deterministic servers and database from Tasks 1–2.
- Produces: zero failures in the Chromium E2E project.

- [x] **Step 1: Run the complete Chromium project and capture the exact remaining list**

Use fresh ports and a fresh SQLite database. Classify every failure as stale test contract, fixture defect, or production defect using its error context and current DOM/API response.

- [x] **Step 2: Apply one RED/GREEN cycle per failure group**

For a stale selector, assert the current accessible role/name. For a fixture defect, create data through the same backend under test and store only paths returned by that backend. For a production defect, add or retain the failing E2E assertion before changing application code.

- [x] **Step 3: Run static verification**

Run targeted ESLint for changed frontend files, `npm run build`, and `git diff --check`.

- [x] **Step 4: Run final regression verification**

Run the full Chromium suite against fresh ports and a fresh SQLite database. Expected: all tests pass, zero did-not-run tests, exit code 0.
