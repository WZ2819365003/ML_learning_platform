// Helpers for capturing page console + failed-network + 4xx/5xx responses.

function attachPageObservers(page) {
  const consoleErrors = [];
  const consoleWarnings = [];
  const failedRequests = [];
  const badResponses = [];
  const pageErrors = [];

  page.on('console', (msg) => {
    const text = `${msg.text()}`;
    if (msg.type() === 'error') consoleErrors.push(text);
    if (msg.type() === 'warning') consoleWarnings.push(text);
  });
  page.on('pageerror', (err) => {
    pageErrors.push(err.message || String(err));
  });
  page.on('requestfailed', (req) => {
    failedRequests.push({ url: req.url(), reason: req.failure()?.errorText });
  });
  page.on('response', (res) => {
    const status = res.status();
    if (status >= 400) {
      badResponses.push({ url: res.url(), status });
    }
  });

  return { consoleErrors, consoleWarnings, failedRequests, badResponses, pageErrors };
}

function summarize(observers) {
  return {
    consoleErrors: observers.consoleErrors.length,
    consoleWarnings: observers.consoleWarnings.length,
    failedRequests: observers.failedRequests.length,
    badResponses: observers.badResponses.length,
    pageErrors: observers.pageErrors.length,
  };
}

async function attachToReport(testInfo, observers, label = 'page-observers') {
  const payload = {
    summary: summarize(observers),
    consoleErrors: observers.consoleErrors,
    consoleWarnings: observers.consoleWarnings.slice(0, 20),
    pageErrors: observers.pageErrors,
    failedRequests: observers.failedRequests,
    badResponses: observers.badResponses,
  };
  await testInfo.attach(label, {
    body: JSON.stringify(payload, null, 2),
    contentType: 'application/json',
  });
  return payload;
}

module.exports = { attachPageObservers, summarize, attachToReport };
