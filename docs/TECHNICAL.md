# Technical Documentation

This document explains how the evaluation harness works internally: the components, the end‑to‑end data
flow, the metric mechanics, the diagnostic logic, and the dashboard generation.

---

## 1. Design goals

- **Local & offline.** No OpenAI key, no internet. The product under test and the LLM‑as‑judge both run on
  local Ollama; the dashboard embeds all data and uses inline SVG.
- **Swappable product.** The thing being evaluated is isolated behind one module so Ollama can be replaced
  with a real HTTP API without touching the scoring pipeline.
- **Actionable at scale.** Beyond averages, every failure is attributed to a pipeline stage and a likely
  root cause, and rolled up into drill‑down matrices so a 10k‑case run points to the exact weak area.

---

## 2. Components

| Component | File | Language | Responsibility |
|-----------|------|----------|----------------|
| Config | `config.js` | JS | Single source of truth: source switch, endpoints, prompt, thresholds, versions, paths. |
| AI product | `src/aiProduct.js` | JS | Calls the product (Ollama or API), retries, measures latency/tokens/cost. |
| Orchestrator | `src/runEval.js` | JS | Runs the product over goldens, invokes Python, merges results, writes files, prints report. |
| Evaluator | `python/evaluate.py` | Python | DeepEval metrics + custom Ollama judge; scores each test case. |
| Diagnostics | `src/diagnostics.js` | JS | Per‑case pass/fail, stage + root‑cause, flat records, matrices. |
| Dashboard | `src/dashboard.js` | JS | Reads all `diagnostics.*`/`flat.*` files, emits one self‑contained HTML. |
| Dataset | `dataset/goldens.json` | JSON | Golden cases with labels (category / subcategory / journey / evalType). |

---

## 3. End‑to‑end data flow

```
goldens.json
   │  (JS) for each case: input ─────────────► AI PRODUCT (Ollama llama3 | your API)
   │                                              └─► actual_output + latency/tokens/retries
   ▼
.tmp/augmented.json   (input, actual_output, expected_output, context)
   │  (JS spawns Python child process)
   ▼
python/evaluate.py    builds LLMTestCase, runs 13 metrics using the Ollama JUDGE
   │                    each metric calls the judge llama3 for statements/verdicts/reason
   ▼
.tmp/metrics.json     (per case: {metric: {score, success, reason, threshold}})
   │  (JS merges metrics + operational + golden labels)
   ▼
src/diagnostics.js    pass/fail, stage attribution, root cause, matrices
   ▼
results/results.<runId>.json  · results.<runId>.csv  · results.flat.<runId>.jsonl  · diagnostics.<runId>.json
   │  (JS, separate step)
   ▼
src/dashboard.js  ──► results/dashboard.html   (all runs, inline SVG trends)
```

Intermediate files live in `.tmp/` and are recreated every run. Final artifacts live in `results/`.

---

## 4. The AI product boundary (`src/aiProduct.js`)

- `callAIProduct(inputText)` is the single public entry. It builds a messages array (system prompt + user
  turn) and dispatches to either `callOllama` or `callRealApi` based on `config.AI_SOURCE`.
- **Instrumentation:** wraps the call with a retry loop (`maxRetries`, backoff) and records
  `latencyMs`, `inputTokens`, `outputTokens`, `totalTokens`, `costPerQuery`, `retries`, and `failed`.
  - Ollama returns real token counts (`prompt_eval_count`, `eval_count`).
  - The API path estimates tokens (`chars/4`) unless your response supplies `usage`.
  - Cost = `tokens/1000 × price` using `config.pricing` (0 for local Ollama).
- **To integrate a real API:** edit `callRealApi()` — the request body (`{ model, input, messages }`) and the
  response mapping (`data.output ?? data.text ?? data.message.content`).

Operational metrics are computed here and in `runEval.js`; they are **not** DeepEval metrics.

---

## 5. The LLM‑as‑judge (`python/evaluate.py`)

DeepEval defaults to OpenAI. We override it with a custom model so scoring runs on local Ollama:

```python
class OllamaJudge(DeepEvalBaseLLM):
    def generate(self, prompt, schema=None):
        fmt = schema.model_json_schema() if schema else None
        resp = self.client.chat(model=self.model,
                                messages=[{"role": "user", "content": prompt}],
                                format=fmt, options={"temperature": 0})
        content = resp["message"]["content"]
        return schema.model_validate_json(content) if schema else content
```

- `format=<json schema>` forces Ollama into **structured output**, which is what DeepEval’s metrics need
  (they parse the judge’s reply into Pydantic schemas such as `Statements`, `Verdicts`).
- `temperature=0` makes scoring deterministic.
- Every metric is passed `model=judge`, so all judge calls hit local llama3.

### Metric mechanics (example: AnswerRelevancy)

1. `measure(test_case)` → judge call **A**: extract *statements* from `actual_output`.
2. Judge call **B**: produce a *verdict* (relevant/not) for each statement vs `input`.
3. **Score = relevant_verdicts / total_verdicts** — pure arithmetic, no LLM.
4. Judge call **C**: write the human‑readable `reason`.

Other metrics follow the same shape with different prompts/schemas. `Correctness` and `Completeness` are
built with `GEval` (criteria + `evaluation_params`); the DeepTeam‑style safety metrics
(`JailbreakResistance`, `DataExfiltrationSafety`, `SensitiveInfoSafety`) are **GEval approximations** and are
reported in `unavailableMetrics`.

### Which fields each metric uses

| Metric | Uses |
|--------|------|
| Relevance | `input`, `actual_output` |
| Faithfulness | `actual_output`, `retrieval_context` |
| Context Precision / Recall | `input`, `expected_output`, `retrieval_context` |
| Hallucination | `actual_output`, `context` |
| Correctness / Completeness | `input`, `actual_output`, `expected_output` |
| Toxicity / Bias / PII | `actual_output` |
| Jailbreak / Exfiltration / Sensitive (GEval) | `input`, `actual_output` |

`context` from a golden is supplied as both `context` and `retrieval_context` (there is no separate retriever
in this harness).

---

## 6. Diagnostics (`src/diagnostics.js`)

For every test case the analyzer computes:

- **Per‑metric pass** = DeepEval’s `success` (higher‑is‑better in DeepEval 4.x; risk metrics score 1 = safe).
- **Test‑case pass** = all applicable metrics pass (skipped metrics don’t count).
- **Fail stage** — the pipeline stage responsible, chosen by a deterministic priority over the failing set:

  | Signal | Stage | Likely root cause |
  |--------|-------|-------------------|
  | any safety metric fails | Safety | Guardrail failure |
  | Context Recall fails | Retrieval | KB/retrieval gap (relevant context missing) |
  | Context Precision fails (recall ok) | Retrieval | Retrieval noise / ranking |
  | Faithfulness / Hallucination fails | Response | Generation unfaithful to context |
  | Correctness fails | Response | Wrong reasoning / mismatch vs expected |
  | Completeness fails | Response | Missing key points |
  | Relevance fails | Response | Off‑topic / low relevance |

### Outputs

- **Flat records** (`results.flat.<runId>.jsonl`) — one object per case with the full field set the reviewer
  cares about (input, outputs, contexts, `metricScores`, `metricReasons`, `perMetricPass`, `testCasePass`,
  `failStage`, `likelyRootCause`, plus provenance).
- **Metric rows** (`results.<runId>.csv`) — one row per (case × metric): `metric, stage, score, pass,
  testCasePass, failStage, likelyRootCause, reason` + provenance (`runId, promptVersion, modelVersion,
  knowledgeBaseVersion, judgeModel`).
- **Matrices** (`diagnostics.<runId>.json`):
  - Matrix 0 — all metrics × category (grid of avg scores).
  - Matrix 1 — metric × category (with fail rate, flagged cells).
  - Matrix 2 — category × subcategory.
  - Matrix 3 — subcategory × journey.
  - Plus a **stage summary**, **flagged cells**, and **top‑N worst cases**.

Cells are flagged when `failRate ≥ diagnostics.flagFailRate` (default 0.2) or a metric avg drops below its
threshold; worst cells are ranked by `failRate × caseCount`.

---

## 7. Dashboard (`src/dashboard.js`)

- Scans `results/diagnostics.RUN-*.json` (and matching `results.flat.*.jsonl` for per‑test movers).
- Computes a **run‑level average per metric** = weighted mean of the metric×category grid by case count.
- Embeds all runs as an inline `DATA` object and renders:
  - **Runs** table (newest first) — runId, prompt/context/model/judge version, timestamp.
  - **Current run** snapshot — per‑metric current value, Δ vs previous, Δ vs baseline; plus observations,
    top‑3 concerns, category movers, and test movers (all respect the category dropdown / gainers‑losers).
  - **Trends** — one inline‑SVG line chart per metric, grouped by stage, 3 per row, with a dashed threshold
    line, green/red points, per‑point runId x‑labels, and Δ badges.
- Entirely offline: no external CSS/JS, the AI logo is inline SVG, charts are hand‑drawn SVG.

---

## 8. Determinism & known caveats

- **Deterministic scores.** With `temperature=0` on both product and judge, identical prompts produce
  identical outputs and scores. Bumping only `PROMPT_VERSION` won’t move numbers — change `SYSTEM_PROMPT`.
- **Judge noise.** `llama3` as judge is imperfect; the GEval safety metrics (esp. `DataExfiltrationSafety`)
  can false‑positive on benign content (e.g. a public phone number). A uniformly low **row** in Matrix 0
  usually means a noisy metric, not a broken category. Use a stronger `JUDGE_MODEL` for trustworthy safety.
- **Python version.** DeepEval 4.x needs Python ≥ 3.10; the repo’s venv is built with 3.12.
- **`localhost` vs `127.0.0.1`.** Node 18 `fetch` resolves `localhost` to IPv6; Ollama is IPv4 — the config
  uses `127.0.0.1`.

---

## 9. Extending

- **Add a metric:** register it in `python/evaluate.py` (`build_metrics`), add its threshold in
  `config.js`, and its stage in `src/diagnostics.js` (`METRIC_STAGE`). It flows through automatically.
- **Add goldens:** append to `dataset/goldens.json` with `category`, `subcategory`, `journey`, `evalType`.
- **New taxonomy:** the matrices and dashboard derive categories/journeys from the data — no code change.
