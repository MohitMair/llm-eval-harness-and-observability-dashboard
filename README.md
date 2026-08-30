# LLM Evaluation Harness using DeepEval

An offline, local evaluation harness for a conversational AI product. (a public website has been used as an example for sample scripts. No sensitive data of IP is compromised)

A **JavaScript orchestrator** drives the flow and calls **Python DeepEval** to score each answer with an
**LLM‑as‑judge** (local Ollama `llama3`). It produces per‑run reports, a diagnostic drill‑down, and a
single self‑contained **HTML trend dashboard**.

- **Product under test:** local Ollama today, your real API later — flip one switch.
- **Judge:** local Ollama (`llama3`) via a custom DeepEval model wrapper — no OpenAI key, no internet. Can be replaced by real LLM config.
- **Metrics:** 13 quality/retrieval/safety metrics + operational metrics (latency, tokens, cost, failure/retry).
- **Diagnostics:** root‑cause tagging + drill‑down matrices (metric × category × subcategory × journey).
- **Dashboard:** one HTML file, inline SVG charts, works by double‑clicking (no CDN/assets).

---

## 1. Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Node.js | ≥ 18 | Uses the built‑in global `fetch`. |
| Python | ≥ 3.10 | DeepEval 4.x requires 3.10+ (3.9 will fail). |
| Ollama | running locally | With the `llama3` model pulled. |

Start Ollama and pull the model:

```bash
ollama pull llama3
# Ollama serves on http://127.0.0.1:11434
```

> Use `127.0.0.1`, not `localhost` — Node’s `fetch` resolves `localhost` to IPv6 `::1`, but Ollama listens on IPv4.

---

## 2. Setup

```bash
# from the project root
python3.12 -m venv python/.venv                 # Python 3.10+ interpreter
./python/.venv/bin/pip install -r python/requirements.txt
```

Node has no runtime dependencies to install. The orchestrator auto‑detects and uses `python/.venv`.

---

## 3. Run an evaluation

```bash
npm run eval            # evaluate the whole golden dataset with local Ollama
```

Useful variants:

```bash
LIMIT=3 npm run eval                 # only the first 3 golden cases (fast smoke test)
AI_SOURCE=api npm run eval           # evaluate your real product API instead of Ollama
PROMPT_VERSION=prompt-v2 SYSTEM_PROMPT="...new prompt..." npm run eval   # A/B a prompt
```

Each run writes four files into `results/`, stamped with a `runId` (`RUN-YYYYMMDD-HHmmss`):

| File | Contents |
|------|----------|
| `results.<runId>.json` | Run header + per‑metric summary + per‑case detail. |
| `results.<runId>.csv` | One row per (test case × metric) — pivot‑friendly. |
| `results.flat.<runId>.jsonl` | One JSON object per test case (all fields, reasons, root cause). |
| `diagnostics.<runId>.json` | Matrices, flagged cells, stage summary, worst cases. |

The console also prints a summary, the diagnostic matrices, and the top failing cases.

---

## 4. Build the trend dashboard

```bash
npm run dashboard       # scans results/ and writes results/dashboard.html
```

Open `results/dashboard.html` in any browser (double‑click). It shows:

- **Runs** table (runId, prompt/model/context/judge version, timestamp).
- **Current run** snapshot: per‑metric value + Δ vs previous + Δ vs baseline, plus an observations panel
  with **top‑3 concerns**, **category movers**, and **test movers**.
- **Trends**: one line chart per metric, grouped into Response / Retrieval / Safety, across all runs.

Re‑run `npm run dashboard` after each `npm run eval` to append the new run.

---

## 5. Switching to your real product API

Everything about "which AI is being tested" lives in one file: [`src/aiProduct.js`](src/aiProduct.js).

1. Set `AI_SOURCE=api` (env or [`config.js`](config.js)).
2. Fill in `api.url` / `api.apiKey` / `api.model` in [`config.js`](config.js) (or via `AI_API_URL`, `AI_API_KEY`, `AI_API_MODEL`).
3. Adjust the request body and response mapping in `callRealApi()` to match your API contract.

The judge stays on local Ollama regardless, so scoring is unchanged.

---

## 6. Metrics

| Group | Metrics | Source |
|-------|---------|--------|
| **Response** | Relevance, Faithfulness, Hallucination, Correctness, Completeness | DeepEval native + GEval (Correctness/Completeness) |
| **Retrieval** | Context Precision, Context Recall | DeepEval native |
| **Safety** | Toxicity, Bias, PII Leakage, Jailbreak Resistance, Data‑Exfiltration Safety, Sensitive‑Info Safety | DeepEval native + GEval approximations |
| **Operational** | Latency, Token usage, Cost/query, Failure rate, Retry rate | Measured in the JS harness (not DeepEval metrics) |

> Jailbreak / Data‑Exfiltration / Sensitive‑Info are **GEval approximations** — DeepEval core has no such
> metric (they belong to DeepTeam). They’re flagged as such in every run’s `unavailableMetrics` note.

Thresholds are configurable in [`config.js`](config.js) under `thresholds`.

---

## 7. Configuration cheat‑sheet

All overridable via environment variables (see [`config.js`](config.js)):

| Env var | Purpose | Default |
|---------|---------|---------|
| `AI_SOURCE` | `ollama` or `api` | `ollama` |
| `OLLAMA_URL` / `OLLAMA_MODEL` | product Ollama endpoint/model | `http://127.0.0.1:11434` / `llama3` |
| `JUDGE_URL` / `JUDGE_MODEL` | judge Ollama endpoint/model | `http://127.0.0.1:11434` / `llama3` |
| `SYSTEM_PROMPT` | product system prompt (for A/B) | built‑in AA prompt |
| `PROMPT_VERSION` / `MODEL_VERSION` / `KB_VERSION` | provenance labels stamped on the run | `prompt-v1` / `llama3` / `kb-v1` |
| `RUN_ID` | run identifier / filename stamp | `RUN-<timestamp>` |
| `LIMIT` | evaluate only the first N goldens | all |
| `PRICE_INPUT_PER_1K` / `PRICE_OUTPUT_PER_1K` | token pricing for cost metric | `0` |

---

## 8. Project layout

```
deepEval/
├── config.js                 # all configuration + the ollama/api switch
├── package.json              # npm scripts: eval, dashboard
├── dataset/goldens.json      # golden test cases (input, expected_output, context, labels)
├── python/
│   ├── evaluate.py           # DeepEval metrics + custom Ollama judge
│   └── requirements.txt
├── src/
│   ├── aiProduct.js          # the AI product under test (ollama/api) + operational metrics
│   ├── runEval.js            # orchestrator: run product → Python → merge → write
│   ├── diagnostics.js        # root‑cause tagging + drill‑down matrices
│   └── dashboard.js          # single‑file HTML dashboard generator
├── docs/
│   ├── TECHNICAL.md          # technical deep‑dive
│   └── ARCHITECTURE.md       # architecture diagrams
└── results/                  # generated per‑run outputs + dashboard (gitignored)
```

See [`docs/TECHNICAL.md`](docs/TECHNICAL.md) for internals and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for diagrams.
