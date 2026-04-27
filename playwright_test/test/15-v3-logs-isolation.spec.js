// 15 — V3 native logs isolation gate.
//
// v3.3.0 decoupled inspector logs from the legacy `training_tasks` table by
// adding `experiment_run_logs` (FK to experiment_runs, with CASCADE).
// `_execute_single_trial` now mirrors legacy logs into this V3-native table
// after every trial, and the inspector reads V3 first.
//
// REGRESSION GUARD: this spec runs a real V3 baseline run, wipes the legacy
// `training_tasks` table (which CASCADE-clears `training_logs`), and asserts
// the inspector STILL returns logs for the run — proving V3-native is the
// authoritative source.
//
// Failure here = the v3.3.0 isolation regressed.
const { test, expect } = require('@playwright/test');
const { execSync } = require('child_process');
const { getJson } = require('../helpers/api');
const { runBaselineFlow } = require('../helpers/v3-flow');

test.describe.configure({ mode: 'serial' });

function mysqlExec(sql) {
  // Run SQL inside the MySQL container.  Tests run on the host; we shell out
  // because spinning up a Python aiomysql client just for one DELETE is
  // overkill and fragile.
  return execSync(
    `docker exec ml_platform_mysql mysql -uroot -p123456 ml_platform -N -e "${sql.replace(/"/g, '\\"')}" 2>/dev/null`,
    { encoding: 'utf8' },
  ).trim();
}

test('15.1 inspector returns logs even after DELETE FROM training_tasks', async ({ request }) => {
  test.setTimeout(180_000);

  // 1. Run a real V3 baseline trial.  The mirror in _execute_single_trial
  //    will copy training_logs → experiment_run_logs once it completes.
  const { run } = await runBaselineFlow(request, { namePrefix: 'v3-logs-iso' });
  const runId = run.run_id || run.id;
  expect(runId).toBeTruthy();

  // 2. Confirm V3-native table actually got rows for this run BEFORE we
  //    touch anything.  If this is 0 the mirror never ran — fail fast.
  const v3CountBefore = parseInt(
    mysqlExec(`SELECT COUNT(*) FROM experiment_run_logs WHERE run_id='${runId}';`),
    10,
  );
  expect(v3CountBefore, `mirror did not populate experiment_run_logs for run ${runId}`)
    .toBeGreaterThan(0);

  // Inspector should see the logs (this part already worked in v3.2.x via the
  // legacy chain — sanity check before we wipe).
  const insBefore = await getJson(request, `/platform/runs/${runId}/inspector`);
  expect(insBefore.ok).toBeTruthy();
  expect(insBefore.body?.logs?.length || 0).toBeGreaterThan(0);

  // 3. DELETE FROM training_tasks — the realistic cleanup that used to break
  //    inspector.  CASCADE wipes training_logs.
  mysqlExec('DELETE FROM training_tasks;');
  const legacyAfter = parseInt(mysqlExec('SELECT COUNT(*) FROM training_logs;'), 10);
  expect(legacyAfter, 'training_logs not actually empty after wipe').toBe(0);

  // 4. The V3-native table must NOT have been touched.
  const v3CountAfter = parseInt(
    mysqlExec(`SELECT COUNT(*) FROM experiment_run_logs WHERE run_id='${runId}';`),
    10,
  );
  expect(v3CountAfter, 'experiment_run_logs got wiped — FK chain bug').toBe(v3CountBefore);

  // 5. The actual regression check: inspector still has logs.
  const insAfter = await getJson(request, `/platform/runs/${runId}/inspector`);
  expect(insAfter.ok, `inspector failed: ${insAfter.status}`).toBeTruthy();
  const logsAfter = insAfter.body?.logs || [];
  expect(logsAfter.length, 'INSPECTOR REGRESSION — V3 native logs not surfaced after legacy wipe')
    .toBeGreaterThan(0);
  expect(insAfter.body?.log_task_id, 'log_task_id should be the run_id when sourced from V3 native')
    .toBe(runId);
});
