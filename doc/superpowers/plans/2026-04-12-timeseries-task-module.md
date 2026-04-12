# TimeSeries Task Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the current TimesFM area into an independent time-series task module with paginated list/new/detail pages and a deployable TimesFM tab inside model deployment.

**Architecture:** Add a dedicated `TimeSeriesDeployment` backend model and `ts` task/deployment routes, keep old `/api/timesfm/*` endpoints as compatibility wrappers, then reshape the existing TS React pages into `tasks list / create / detail` while preserving professional metadata exposure.

**Tech Stack:** FastAPI, SQLAlchemy async, SQLite, React 18, React Router, Ant Design 5, ECharts, Playwright, pytest

---

### Task 1: Add failing backend tests for time-series deployments and tasks

**Files:**
- Create: `ml_platform/tests/test_timeseries_routes.py`
- Modify: `ml_platform/tests/conftest.py` or local fixture pattern inside the new test file if needed
- Test: `ml_platform/tests/test_timeseries_routes.py`

- [ ] **Step 1: Write failing tests**

Cover:

- create/list/pause/delete a `TimeSeriesDeployment`
- create a task bound to `deployment_id`
- old `/api/timesfm/start` maps to the new service layer

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ml_platform && python -m pytest tests/test_timeseries_routes.py -q`

- [ ] **Step 3: Implement minimal backend models and routes**

Touch:

- `ml_platform/app/models/database.py`
- `ml_platform/app/models/schemas.py`
- `ml_platform/app/api/routes/timesfm.py`
- new `ml_platform/app/services/timeseries_service.py`

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ml_platform && python -m pytest tests/test_timeseries_routes.py -q`

- [ ] **Step 5: Commit**

`git commit -m "feat: add time-series task and deployment routes"`

### Task 2: Add startup schema patching for existing SQLite databases

**Files:**
- Modify: `ml_platform/app/main.py`
- Modify: `ml_platform/app/models/database.py`
- Test: `ml_platform/tests/test_timeseries_routes.py`

- [ ] **Step 1: Write failing test or fixture scenario**

Simulate an older DB missing new TS columns/tables and confirm startup patching succeeds.

- [ ] **Step 2: Run targeted test and watch it fail**

- [ ] **Step 3: Implement schema patch helper**

Add a small startup-time helper that:

- creates new tables
- adds missing `ts_forecast_tasks` columns if absent

- [ ] **Step 4: Re-run tests**

- [ ] **Step 5: Commit**

`git commit -m "feat: patch legacy sqlite schema for time-series module"`

### Task 3: Expand frontend API client and routing

**Files:**
- Modify: `ml_platform_web/src/services/api.js`
- Modify: `ml_platform_web/src/App.jsx`
- Modify: `ml_platform_web/src/components/layout/Sidebar.jsx`
- Test: `tests/timeseries-task-flow.spec.js`

- [ ] **Step 1: Write failing Playwright smoke test**

Cover:

- `/ts/tasks`
- `/ts/tasks/new`
- `/ts/tasks/:id`
- old route redirect behavior

- [ ] **Step 2: Run Playwright test and verify it fails**

- [ ] **Step 3: Implement API client and route wiring**

Add `tsApi` task/deployment calls and redirects from old routes.

- [ ] **Step 4: Re-run the Playwright test**

- [ ] **Step 5: Commit**

`git commit -m "feat: wire time-series task routes and api client"`

### Task 4: Rebuild the time-series create page

**Files:**
- Modify: `ml_platform_web/src/pages/TSConfig.jsx`
- Modify: `ml_platform_web/src/styles/global.css` if shared TS styles are needed
- Test: `tests/timeseries-task-flow.spec.js`

- [ ] **Step 1: Extend the failing Playwright flow**

Assert:

- no preload/model-status console UI
- dataset + deployment selection required
- right-side professional summary card present

- [ ] **Step 2: Run the test and verify it fails**

- [ ] **Step 3: Implement the page**

Convert `TSConfig` into the new create page while keeping the filename for compatibility.

- [ ] **Step 4: Re-run test**

- [ ] **Step 5: Commit**

`git commit -m "feat: redesign time-series task creation page"`

### Task 5: Rebuild the time-series list and detail pages

**Files:**
- Modify: `ml_platform_web/src/pages/TSMonitor.jsx`
- Modify: `ml_platform_web/src/pages/TSResults.jsx`
- Test: `tests/timeseries-task-flow.spec.js`

- [ ] **Step 1: Extend the failing Playwright flow**

Assert:

- paginated list exists
- click row enters detail
- detail page shows deployment info, URL, params, raw response
- back returns to list

- [ ] **Step 2: Run the test and verify it fails**

- [ ] **Step 3: Implement list/detail UI**

Use the existing files as wrappers around the new task list/detail structure.

- [ ] **Step 4: Re-run test**

- [ ] **Step 5: Commit**

`git commit -m "feat: redesign time-series task list and detail views"`

### Task 6: Implement the TimesFM deployment tab in model deployment

**Files:**
- Modify: `ml_platform_web/src/pages/ModelDeploy.jsx`
- Modify: `ml_platform/app/api/routes/timesfm.py`
- Modify: `ml_platform/app/models/schemas.py`
- Test: `tests/timeseries-deployments.spec.js`

- [ ] **Step 1: Write failing UI/backend coverage**

Cover:

- TimesFM tab is no longer empty
- can create deployment
- can copy URL
- can pause/resume deployment

- [ ] **Step 2: Run test and verify it fails**

- [ ] **Step 3: Implement the TimesFM deployment tab**

Preserve ML/DL tabs and add the same “list + detail component” quality level for TimesFM.

- [ ] **Step 4: Re-run tests**

- [ ] **Step 5: Commit**

`git commit -m "feat: add timesfm deployment management tab"`

### Task 7: Tighten model-management messaging and pagination constraints

**Files:**
- Modify: `ml_platform_web/src/pages/ModelManagement.jsx`
- Modify: `ml_platform/app/api/routes/model_mgmt.py`
- Test: `ml_platform/tests/test_unified_model_assets.py`

- [ ] **Step 1: Write failing regression test**

Cover:

- `/api/models/list?page_size=200` no longer hard-fails for internal UI usage
- universal tab message points users to deployment instead of pretending TimesFM is a model asset

- [ ] **Step 2: Run failing tests**

- [ ] **Step 3: Implement the fix**

- [ ] **Step 4: Re-run tests**

- [ ] **Step 5: Commit**

`git commit -m "fix: align model management with time-series deployment flow"`

### Task 8: Full verification

**Files:**
- Test: `ml_platform/tests/test_timeseries_routes.py`
- Test: `ml_platform/tests/test_unified_model_assets.py`
- Test: `tests/timeseries-task-flow.spec.js`
- Test: `tests/timeseries-deployments.spec.js`

- [ ] **Step 1: Run backend verification**

Run: `cd ml_platform && python -m pytest tests/test_timeseries_routes.py tests/test_unified_model_assets.py -q`

- [ ] **Step 2: Run frontend build**

Run: `cd ml_platform_web && npm run build`

- [ ] **Step 3: Run Playwright verification**

Run:

`npx playwright test tests/timeseries-task-flow.spec.js tests/timeseries-deployments.spec.js --project=chromium --reporter=line`

- [ ] **Step 4: Fix any failures and rerun**

- [ ] **Step 5: Commit final integration**

`git commit -m "feat: ship time-series task module v1"`
