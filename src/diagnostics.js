// Diagnostic layer: turns per-test-case metric scores into actionable pointers.
// - Normalizes every result into flat records (per test case, and per test-case x metric).
// - Attributes each failure to a pipeline STAGE (Retrieval / Response / Safety) and a likely root cause.
// - Builds drill-down matrices: Metric x Category -> Category x Subcategory -> Subcategory x Journey.
// - Flags problem cells and ranks the worst cells and worst cases.

// Which pipeline stage each metric belongs to (all metrics: higher score = better in deepeval 4.x).
const METRIC_STAGE = {
  ContextualPrecision: "Retrieval",
  ContextualRecall: "Retrieval",
  Relevance: "Response",
  Faithfulness: "Response",
  Hallucination: "Response",
  Correctness: "Response",
  Completeness: "Response",
  Toxicity: "Safety",
  Bias: "Safety",
  PIILeakage: "Safety",
  JailbreakResistance: "Safety",
  DataExfiltrationSafety: "Safety",
  SensitiveInfoSafety: "Safety",
};

// Columns shown in the drill-down matrices (kept concise for readability).
const MATRIX_METRICS = [
  "Correctness",
  "Relevance",
  "Faithfulness",
  "Completeness",
  "ContextualPrecision",
  "ContextualRecall",
];

// Full metric set (every metric, in stage order) for the metric x category grid.
const ALL_METRICS = Object.keys(METRIC_STAGE);

const isNum = (x) => typeof x === "number" && !Number.isNaN(x);
const avg = (nums) => {
  const v = nums.filter(isNum);
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
};
const round = (x, d = 4) => (isNum(x) ? Number(x.toFixed(d)) : null);

// Derive per-metric pass/fail, overall test-case pass, fail stage and likely root cause.
function analyzeCase(c) {
  const perMetricPass = {};
  const failedMetrics = [];
  const applicable = [];

  for (const [name, m] of Object.entries(c.metrics)) {
    if (m.success === true || m.success === false) {
      applicable.push(name);
      perMetricPass[name] = m.success;
      if (m.success === false) failedMetrics.push(name);
    } else {
      perMetricPass[name] = null; // skipped / not applicable
    }
  }

  // A test case passes only when every verdict-bearing (applicable) metric meets its threshold.
  const testCasePass = applicable.length > 0 && failedMetrics.length === 0;

  const failedSet = new Set(failedMetrics);
  const fails = (n) => failedSet.has(n);
  const stageOf = (names) => names.find((n) => fails(n));

  let failStage = null;
  let likelyRootCause = "";

  const safetyFailed = failedMetrics.filter((n) => METRIC_STAGE[n] === "Safety");

  if (safetyFailed.length) {
    failStage = "Safety";
    likelyRootCause = `Guardrail failure: ${safetyFailed.join(", ")} below threshold`;
  } else if (fails("ContextualRecall")) {
    failStage = "Retrieval";
    likelyRootCause = "Knowledge/retrieval gap: relevant ground-truth context missing (low context recall)";
  } else if (fails("ContextualPrecision")) {
    failStage = "Retrieval";
    likelyRootCause = "Retrieval noise/ranking: irrelevant context ranked high (low context precision)";
  } else if (fails("Faithfulness") || fails("Hallucination")) {
    failStage = "Response";
    likelyRootCause = "Generation unfaithful to retrieved context (contradiction / hallucination)";
  } else if (fails("Correctness")) {
    failStage = "Response";
    likelyRootCause = "Incorrect reasoning or answer mismatch vs expected output";
  } else if (fails("Completeness")) {
    failStage = "Response";
    likelyRootCause = "Incomplete answer: key expected points missing";
  } else if (fails("Relevance")) {
    failStage = "Response";
    likelyRootCause = "Off-topic / low-relevance response (padding or drift)";
  }

  return { perMetricPass, failedMetrics, testCasePass, failStage, likelyRootCause };
}

// One JSON object per test case (written to results.flat.jsonl).
function buildCaseRecords(cases, versions) {
  return cases.map((c) => {
    const a = analyzeCase(c);
    const metricScores = {};
    const metricReasons = {};
    for (const [name, m] of Object.entries(c.metrics)) {
      metricScores[name] = m.score;
      metricReasons[name] = m.reason;
    }
    return {
      ...versions,
      testCaseId: c.id,
      journey: c.journey,
      category: c.category,
      subcategory: c.subcategory,
      evalType: c.evalType,
      userInput: c.input,
      expectedOutput: c.expected_output,
      actualOutput: c.actual_output,
      retrievedContext: c.retrievedContext || [],
      expectedContext: c.expectedContext || [],
      metricScores,
      metricReasons,
      perMetricPass: a.perMetricPass,
      testCasePass: a.testCasePass,
      failStage: a.failStage,
      failedMetrics: a.failedMetrics,
      likelyRootCause: a.likelyRootCause,
    };
  });
}

// Long/tidy rows: one per (test case x metric). Written to results.csv for pivoting.
function buildMetricRows(caseRecords) {
  const rows = [];
  for (const r of caseRecords) {
    for (const [metric, score] of Object.entries(r.metricScores)) {
      rows.push({
        runId: r.runId,
        promptVersion: r.promptVersion,
        modelVersion: r.modelVersion,
        knowledgeBaseVersion: r.knowledgeBaseVersion,
        judgeModel: r.judgeModel,
        testCaseId: r.testCaseId,
        journey: r.journey,
        category: r.category,
        subcategory: r.subcategory,
        evalType: r.evalType,
        metric,
        stage: METRIC_STAGE[metric] || "Other",
        score,
        pass: r.perMetricPass[metric],
        testCasePass: r.testCasePass,
        failStage: r.failStage || "",
        likelyRootCause: r.likelyRootCause || "",
        reason: r.metricReasons[metric] || "",
      });
    }
  }
  return rows;
}

// Aggregate a set of case records into one matrix cell.
function makeCell(dims, group, thresholds, flagFailRate) {
  const testCases = group.length;
  const failCount = group.filter((r) => !r.testCasePass).length;
  const failRate = testCases ? failCount / testCases : 0;

  const metrics = {};
  const flagReasons = [];
  for (const name of MATRIX_METRICS) {
    const scores = group.map((r) => r.metricScores[name]).filter(isNum);
    const avgScore = round(avg(scores));
    metrics[name] = { avgScore, n: scores.length };
    const th = thresholds[metricThresholdKey(name)];
    if (avgScore != null && isNum(th) && avgScore < th) {
      flagReasons.push(`${name} avg ${avgScore} < ${th}`);
    }
  }
  if (failRate >= flagFailRate) flagReasons.push(`failRate ${(failRate * 100).toFixed(0)}% >= ${(flagFailRate * 100).toFixed(0)}%`);

  return {
    ...dims,
    testCases,
    failCount,
    failRate: round(failRate),
    metrics,
    flagged: flagReasons.length > 0,
    flagReasons,
    // Ranking weight: big + broken areas first.
    severity: round(failRate * testCases, 3),
  };
}

function metricThresholdKey(metricName) {
  const map = {
    Correctness: "correctness",
    Relevance: "relevance",
    Faithfulness: "faithfulness",
    Completeness: "completeness",
    ContextualPrecision: "contextualPrecision",
    ContextualRecall: "contextualRecall",
  };
  return map[metricName];
}

function groupBy(records, keyFn) {
  const map = new Map();
  for (const r of records) {
    const k = keyFn(r);
    if (!map.has(k)) map.set(k, []);
    map.get(k).push(r);
  }
  return map;
}

function buildMatrices(caseRecords, thresholds, flagFailRate) {
  // Matrix 1: Metric x Category
  const m1 = [...groupBy(caseRecords, (r) => r.category)].map(([category, group]) =>
    makeCell({ category }, group, thresholds, flagFailRate)
  );
  // Matrix 2: Category x Subcategory
  const m2 = [...groupBy(caseRecords, (r) => `${r.category}||${r.subcategory}`)].map(([k, group]) => {
    const [category, subcategory] = k.split("||");
    return makeCell({ category, subcategory }, group, thresholds, flagFailRate);
  });
  // Matrix 3: Subcategory x Journey
  const m3 = [...groupBy(caseRecords, (r) => `${r.subcategory}||${r.journey}`)].map(([k, group]) => {
    const [subcategory, journey] = k.split("||");
    return makeCell({ subcategory, journey }, group, thresholds, flagFailRate);
  });

  const sortWorst = (arr) => [...arr].sort((a, b) => b.severity - a.severity || b.failRate - a.failRate);
  return {
    metricByCategory: sortWorst(m1),
    categoryBySubcategory: sortWorst(m2),
    subcategoryByJourney: sortWorst(m3),
  };
}

// Deterministic short codes for category column headers in the metric x category grid.
function categoryCodes(categories) {
  const codes = {};
  const used = new Set();
  for (const cat of categories) {
    const letters = (cat.match(/[A-Za-z]/g) || []).join("").toUpperCase();
    let code = letters.slice(0, 3) || "CAT";
    let i = 1;
    while (used.has(code)) code = (letters.slice(0, 2) + i++).toUpperCase();
    used.add(code);
    codes[cat] = code;
  }
  return codes;
}

// Full grid: every eval metric (row) x every business category (column).
// Each cell = avg score + pass rate for that metric within that category.
function buildMetricByCategoryGrid(caseRecords) {
  const categories = [...new Set(caseRecords.map((r) => r.category))].sort();
  const metrics = ALL_METRICS.filter((m) => caseRecords.some((r) => m in r.metricScores));

  const rows = metrics.map((metric) => {
    const byCategory = {};
    for (const cat of categories) {
      const group = caseRecords.filter((r) => r.category === cat);
      const scores = group.map((r) => r.metricScores[metric]).filter(isNum);
      const decided = group.map((r) => r.perMetricPass[metric]).filter((v) => v === true || v === false);
      const passed = decided.filter((v) => v === true).length;
      byCategory[cat] = {
        avgScore: round(avg(scores)),
        passRate: decided.length ? round(passed / decided.length, 3) : null,
        n: scores.length,
      };
    }
    return { metric, stage: METRIC_STAGE[metric], byCategory };
  });

  return { categories, codes: categoryCodes(categories), metrics, rows };
}

function buildStageSummary(caseRecords) {
  const failing = caseRecords.filter((r) => !r.testCasePass);
  const byStage = { Retrieval: 0, Response: 0, Safety: 0, None: 0 };
  for (const r of failing) byStage[r.failStage || "None"] += 1;
  const total = failing.length || 1;
  return {
    totalCases: caseRecords.length,
    failingCases: failing.length,
    byStage,
    byStagePct: Object.fromEntries(
      Object.entries(byStage).map(([k, v]) => [k, round(v / total, 3)])
    ),
  };
}

function rankWorstCases(caseRecords, topN) {
  return caseRecords
    .filter((r) => !r.testCasePass)
    .map((r) => ({
      testCaseId: r.testCaseId,
      category: r.category,
      subcategory: r.subcategory,
      journey: r.journey,
      failStage: r.failStage,
      failedMetrics: r.failedMetrics,
      likelyRootCause: r.likelyRootCause,
      lowestScore: round(Math.min(...Object.values(r.metricScores).filter(isNum))),
      failReasons: Object.fromEntries(r.failedMetrics.map((m) => [m, r.metricReasons[m]])),
    }))
    .sort((a, b) => b.failedMetrics.length - a.failedMetrics.length || a.lowestScore - b.lowestScore)
    .slice(0, topN);
}

// Attach the top-N worst test cases to each flagged matrix-1 (category) cell.
function attachWorstCasesToCells(cells, caseRecords, keyName, topN) {
  return cells.map((cell) => {
    if (!cell.flagged) return cell;
    const inCell = caseRecords.filter((r) => r[keyName] === cell[keyName] && !r.testCasePass);
    const worst = inCell
      .sort((a, b) => b.failedMetrics.length - a.failedMetrics.length)
      .slice(0, topN)
      .map((r) => ({ testCaseId: r.testCaseId, failStage: r.failStage, failedMetrics: r.failedMetrics, likelyRootCause: r.likelyRootCause }));
    return { ...cell, worstCases: worst };
  });
}

function buildDiagnostics(cases, versions, config) {
  const { thresholds } = config;
  const flagFailRate = config.diagnostics.flagFailRate;
  const topCells = config.diagnostics.topWorstCells;
  const topCases = config.diagnostics.topWorstCases;

  const caseRecords = buildCaseRecords(cases, versions);
  const metricRows = buildMetricRows(caseRecords);
  const matrices = buildMatrices(caseRecords, thresholds, flagFailRate);
  matrices.metricByCategory = attachWorstCasesToCells(matrices.metricByCategory, caseRecords, "category", topCases);

  const diagnostics = {
    ...versions,
    generatedAt: new Date().toISOString(),
    stageSummary: buildStageSummary(caseRecords),
    metricByCategoryGrid: buildMetricByCategoryGrid(caseRecords),
    matrices,
    flaggedCells: {
      metricByCategory: matrices.metricByCategory.filter((c) => c.flagged).slice(0, topCells),
      categoryBySubcategory: matrices.categoryBySubcategory.filter((c) => c.flagged).slice(0, topCells),
      subcategoryByJourney: matrices.subcategoryByJourney.filter((c) => c.flagged).slice(0, topCells),
    },
    worstCases: rankWorstCases(caseRecords, topCases),
  };

  return { caseRecords, metricRows, diagnostics };
}

// --- Serialization helpers ------------------------------------------------
function toJsonl(records) {
  return records.map((r) => JSON.stringify(r)).join("\n") + "\n";
}

function csvEscape(v) {
  if (v == null) return "";
  let s = Array.isArray(v) ? v.join(" | ") : String(v);
  if (/[",\n]/.test(s)) s = `"${s.replace(/"/g, '""')}"`;
  return s;
}

function toCsv(rows) {
  if (!rows.length) return "";
  const cols = Object.keys(rows[0]);
  const header = cols.join(",");
  const body = rows.map((r) => cols.map((c) => csvEscape(r[c])).join(",")).join("\n");
  return `${header}\n${body}\n`;
}

// --- Console rendering ----------------------------------------------------
function printDiagnostics(diagnostics) {
  const line = "-".repeat(72);
  const flag = (c) => (c.flagged ? "\u{1F534}" : "  ");
  const pct = (x) => `${(x * 100).toFixed(0)}%`;

  console.log(`\n${line}`);
  console.log("  DIAGNOSTIC MATRICES");
  console.log(line);

  const ss = diagnostics.stageSummary;
  console.log(`\n  Failing cases: ${ss.failingCases}/${ss.totalCases}  ` +
    `| by stage -> Retrieval ${pct(ss.byStagePct.Retrieval)}, Response ${pct(ss.byStagePct.Response)}, Safety ${pct(ss.byStagePct.Safety)}`);

  // All-metrics x category grid (metric rows, category columns).
  const grid = diagnostics.metricByCategoryGrid;
  if (grid && grid.categories.length) {
    console.log(`\n  Matrix 0 - All eval metrics x Business Category (avg score)`);
    const codes = grid.categories.map((c) => grid.codes[c]);
    console.log("   " + "Metric".padEnd(22) + "Stage".padEnd(11) + codes.map((c) => c.padStart(8)).join(""));
    for (const row of grid.rows) {
      const cells = grid.categories.map((c) => {
        const s = row.byCategory[c].avgScore;
        return (s == null ? "-" : s.toFixed(2)).padStart(8);
      }).join("");
      console.log("   " + row.metric.padEnd(22) + row.stage.padEnd(11) + cells);
    }
    console.log("   Legend: " + grid.categories.map((c) => `${grid.codes[c]}=${c}`).join("  |  "));
  }

  const printMatrix = (title, cells, dimCols) => {
    console.log(`\n  ${title}`);
    const head =
      "   " +
      dimCols.map((d) => d.label.padEnd(d.w)).join("") +
      "cases  fail   " +
      MATRIX_METRICS.map((m) => m.slice(0, 8).padStart(9)).join("");
    console.log(head);
    for (const c of cells.slice(0, 15)) {
      const dims = dimCols.map((d) => String(c[d.key] || "").slice(0, d.w - 1).padEnd(d.w)).join("");
      const scores = MATRIX_METRICS.map((m) => {
        const s = c.metrics[m].avgScore;
        return (s == null ? "-" : s.toFixed(2)).padStart(9);
      }).join("");
      console.log(` ${flag(c)}${dims}${String(c.testCases).padEnd(7)}${pct(c.failRate).padEnd(7)}${scores}`);
    }
  };

  printMatrix("Matrix 1 - Metric x Business Category", diagnostics.matrices.metricByCategory, [
    { key: "category", label: "Category", w: 34 },
  ]);
  printMatrix("Matrix 2 - Category x Subcategory (worst first)", diagnostics.matrices.categoryBySubcategory, [
    { key: "category", label: "Category", w: 26 },
    { key: "subcategory", label: "Subcategory", w: 26 },
  ]);
  printMatrix("Matrix 3 - Subcategory x Journey (worst first)", diagnostics.matrices.subcategoryByJourney, [
    { key: "subcategory", label: "Subcategory", w: 26 },
    { key: "journey", label: "Journey", w: 24 },
  ]);

  console.log(`\n  Worst test cases (by # failed metrics):`);
  for (const w of diagnostics.worstCases) {
    console.log(`   - ${w.testCaseId} [${w.category} / ${w.subcategory}] stage=${w.failStage} ` +
      `fails=${w.failedMetrics.join(",")}`);
    console.log(`       root cause: ${w.likelyRootCause}`);
  }
  console.log(`\n${line}\n`);
}

module.exports = { buildDiagnostics, toJsonl, toCsv, printDiagnostics, METRIC_STAGE, MATRIX_METRICS };
