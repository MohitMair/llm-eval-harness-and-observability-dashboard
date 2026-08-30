// JS orchestrator:
//   1. Load goldens
//   2. Call the AI product (Ollama or your API) to get actual outputs + operational metrics
//   3. Hand the augmented dataset to Python DeepEval for quality/safety metrics
//   4. Merge everything, print a summary, write results.json

const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");
const config = require("../config");
const { callAIProduct } = require("./aiProduct");
const { buildDiagnostics, toJsonl, toCsv, printDiagnostics } = require("./diagnostics");

const ROOT = path.resolve(__dirname, "..");
const p = (rel) => path.join(ROOT, rel);

function ensureDir(filePath) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
}

function loadGoldens() {
  const raw = fs.readFileSync(p(config.paths.goldens), "utf8");
  return JSON.parse(raw);
}
// Run the Python DeepEval evaluator as a child process.
function runPythonEval(augmentedPath, metricsOutPath) {
  return new Promise((resolve, reject) => {
    const venvPython = p("python/.venv/bin/python");
    const pythonBin = process.env.PYTHON_BIN || (fs.existsSync(venvPython) ? venvPython : "python3");
    const args = [
      p("python/evaluate.py"),
      "--input", augmentedPath,
      "--output", metricsOutPath,
    ];
    const env = {
      ...process.env,
      JUDGE_URL: config.judge.baseUrl,
      JUDGE_MODEL: config.judge.model,
      THRESHOLDS: JSON.stringify(config.thresholds),
      DEEPEVAL_TELEMETRY_OPT_OUT: "YES",
      DEEPEVAL_DISABLE_PROGRESS_BAR: "YES",
    };
    const child = spawn(pythonBin, args, { stdio: "inherit", env });
    child.on("error", reject);
    child.on("close", (code) =>
      code === 0 ? resolve() : reject(new Error(`Python evaluator exited with code ${code}`))
    );
  });
}

function avg(nums) {
  const v = nums.filter((n) => typeof n === "number");
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : 0;
}

function percentile(nums, pct) {
  const v = nums.filter((n) => typeof n === "number").sort((a, b) => a - b);
  if (!v.length) return 0;
  const idx = Math.min(v.length - 1, Math.ceil((pct / 100) * v.length) - 1);
  return v[idx];
}

function summariseOperational(ops) {
  const total = ops.length;
  const failures = ops.filter((o) => o.failed).length;
  const withRetry = ops.filter((o) => o.retries > 0).length;
  const totalAttempts = ops.reduce((a, o) => a + 1 + o.retries, 0);
  const totalRetries = ops.reduce((a, o) => a + o.retries, 0);
  const latencies = ops.map((o) => o.latencyMs);
  return {
    queries: total,
    latency: {
      avgMs: Math.round(avg(latencies)),
      p95Ms: Math.round(percentile(latencies, 95)),
      maxMs: Math.max(...latencies, 0),
      slaMs: config.latencySlaMs,
      withinSlaRate: total ? ops.filter((o) => o.latencyMs <= config.latencySlaMs).length / total : 0,
    },
    tokens: {
      avgInput: Math.round(avg(ops.map((o) => o.inputTokens))),
      avgOutput: Math.round(avg(ops.map((o) => o.outputTokens))),
      avgTotal: Math.round(avg(ops.map((o) => o.totalTokens))),
      grandTotal: ops.reduce((a, o) => a + o.totalTokens, 0),
    },
    cost: {
      currency: config.pricing.currency,
      avgPerQuery: Number(avg(ops.map((o) => o.costPerQuery)).toFixed(6)),
      total: Number(ops.reduce((a, o) => a + o.costPerQuery, 0).toFixed(6)),
    },
    reliability: {
      failureRate: total ? failures / total : 0,
      successRate: total ? (total - failures) / total : 0,
      retryRate: total ? withRetry / total : 0, // fraction of queries that needed >=1 retry
      retriesPerQuery: total ? totalRetries / total : 0,
      totalAttempts,
    },
  };
}

function fmtPct(x) {
  return `${(x * 100).toFixed(1)}%`;
}

function printSummary(final) {
  const line = "-".repeat(64);
  console.log(`\n${line}`);
  console.log(`  DeepEval report  |  AI_SOURCE = ${config.AI_SOURCE}`);
  console.log(line);

  // Quality/safety aggregate table.
  console.log("\n  QUALITY & SAFETY METRICS (avg score across cases)");
  const agg = final.qualitySummary;
  Object.keys(agg).forEach((name) => {
    const m = agg[name];
    const score = m.avgScore == null ? "n/a" : m.avgScore.toFixed(3);
    const pass = `${m.passed}/${m.count}`;
    const note = m.available === false ? "  (NOT AVAILABLE in installed deepeval)" : "";
    console.log(`   - ${name.padEnd(26)} score=${String(score).padEnd(7)} pass=${pass}${note}`);
  });

  // Operational table.
  const o = final.operationalSummary;
  console.log("\n  OPERATIONAL / SUCCESS-RATE METRICS");
  console.log(`   - Latency avg/p95/max     ${o.latency.avgMs}ms / ${o.latency.p95Ms}ms / ${o.latency.maxMs}ms`);
  console.log(`   - Within SLA (${o.latency.slaMs}ms)     ${fmtPct(o.latency.withinSlaRate)}`);
  console.log(`   - Tokens avg in/out/total  ${o.tokens.avgInput} / ${o.tokens.avgOutput} / ${o.tokens.avgTotal}`);
  console.log(`   - Cost per query (avg)     ${o.cost.avgPerQuery} ${o.cost.currency}`);
  console.log(`   - Success rate            ${fmtPct(o.reliability.successRate)}`);
  console.log(`   - Failure rate            ${fmtPct(o.reliability.failureRate)}`);
  console.log(`   - Retry rate              ${fmtPct(o.reliability.retryRate)} (avg ${o.reliability.retriesPerQuery.toFixed(2)} retries/query)`);
  console.log(`\n  Full details written to ${config.paths.finalResults}`);
  console.log(`${line}\n`);
}

async function main() {
  let goldens = loadGoldens();
  const limit = Number(process.env.LIMIT || 0);
  if (limit > 0) goldens = goldens.slice(0, limit);
  console.log(`Loaded ${goldens.length} goldens. Calling AI product via "${config.AI_SOURCE}"...`);
  const augmented = [];
  const ops = [];

  for (const g of goldens) {
    process.stdout.write(`  -> ${g.id} `);
    const result = await callAIProduct(g.input);
    console.log(result.failed ? `FAILED (${result.error})` : `ok (${result.latencyMs}ms)`);

    augmented.push({
      id: g.id,
      input: g.input,
      actual_output: result.output,
      expected_output: g.expected_output,
      context: g.context || [],
    });
    ops.push({ id: g.id, ...result });
  }

  // Persist augmented dataset for Python.
  ensureDir(p(config.paths.augmented));
  fs.writeFileSync(p(config.paths.augmented), JSON.stringify(augmented, null, 2));

  console.log("\nRunning DeepEval (Python) with local Ollama as judge...\n");
  ensureDir(p(config.paths.metricsOut));
  await runPythonEval(p(config.paths.augmented), p(config.paths.metricsOut));

  const metrics = JSON.parse(fs.readFileSync(p(config.paths.metricsOut), "utf8"));

  // Build quality summary (avg score + pass counts per metric).
  const qualitySummary = {};
  for (const caseResult of metrics.cases) {
    for (const [name, m] of Object.entries(caseResult.metrics)) {
      if (!qualitySummary[name]) {
        qualitySummary[name] = { count: 0, passed: 0, scores: [], available: m.available !== false };
      }
      const q = qualitySummary[name];
      q.count += 1;
      if (m.success === true) q.passed += 1;
      if (typeof m.score === "number") q.scores.push(m.score);
      if (m.available === false) q.available = false;
    }
  }
  for (const name of Object.keys(qualitySummary)) {
    const q = qualitySummary[name];
    q.avgScore = q.scores.length ? Number(avg(q.scores).toFixed(4)) : null;
    delete q.scores;
  }

  const operationalSummary = summariseOperational(ops);

  // Merge per-case operational + quality data, and re-attach the golden's labels/context.
  const opsById = Object.fromEntries(ops.map((o) => [o.id, o]));
  const goldenById = Object.fromEntries(goldens.map((g) => [g.id, g]));
  const cases = metrics.cases.map((c) => {
    const g = goldenById[c.id] || {};
    return {
      ...c,
      evalType: g.evalType,
      category: g.category,
      subcategory: g.subcategory,
      journey: g.journey,
      retrievedContext: g.context || [],
      expectedContext: g.expected_context || g.context || [],
      operational: (({ output, ...rest }) => rest)(opsById[c.id] || {}),
    };
  });

  // Diagnostic layer: flat records, root-cause tagging, drill-down matrices.
  // Run metadata (incl. the judge model) is stamped onto every record.
  const runMeta = {
    ...config.versions,
    judgeModel: `ollama:${config.judge.model}`,
  };
  const { caseRecords, metricRows, diagnostics } = buildDiagnostics(cases, runMeta, config);

  const final = {
    generatedAt: new Date().toISOString(),
    runId: config.versions.runId,
    aiSource: config.AI_SOURCE,
    versions: config.versions,
    judge: config.judge,
    thresholds: config.thresholds,
    qualitySummary,
    operationalSummary,
    stageSummary: diagnostics.stageSummary,
    unavailableMetrics: metrics.unavailableMetrics || [],
    cases,
  };

  // Stamp runId into every result filename, e.g. results.json -> results.run-123.json
  const stamp = (rel) => rel.replace(/(\.[^.]+)$/, `.${config.versions.runId}$1`);
  const outFinal = stamp(config.paths.finalResults);
  const outJsonl = stamp(config.paths.flatJsonl);
  const outCsv = stamp(config.paths.flatCsv);
  const outDiag = stamp(config.paths.diagnostics);

  ensureDir(p(outFinal));
  fs.writeFileSync(p(outFinal), JSON.stringify(final, null, 2));
  fs.writeFileSync(p(outJsonl), toJsonl(caseRecords));
  fs.writeFileSync(p(outCsv), toCsv(metricRows));
  fs.writeFileSync(p(outDiag), JSON.stringify(diagnostics, null, 2));

  printSummary(final);
  printDiagnostics(diagnostics);

  console.log(`Outputs (runId=${config.versions.runId}): ${outFinal}, ${outDiag}, ${outJsonl}, ${outCsv}\n`);

  if (final.unavailableMetrics.length) {
    console.log("NOTE: these requested metrics are NOT native to your installed deepeval:");
    final.unavailableMetrics.forEach((m) => console.log(`   - ${m}`));
    console.log("");
  }
}

main().catch((err) => {
  console.error("\nEvaluation failed:", err);
  process.exit(1);
});
