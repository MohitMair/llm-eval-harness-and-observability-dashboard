# Architecture

Diagrams for the DeepEval AA‑breakdown evaluation harness. All diagrams are Mermaid and render on GitHub /
VS Code Mermaid preview.

---

## 1. Component / container view

```mermaid
flowchart TB
    subgraph JS["JavaScript orchestrator (Node ≥ 18)"]
        CFG["config.js<br/>source switch · prompt · thresholds · versions"]
        PROD["src/aiProduct.js<br/>product call + operational metrics"]
        RUN["src/runEval.js<br/>orchestrator"]
        DIAG["src/diagnostics.js<br/>root-cause + matrices"]
        DASH["src/dashboard.js<br/>HTML generator"]
    end

    subgraph PY["Python evaluator (venv, 3.10+)"]
        EVAL["python/evaluate.py<br/>DeepEval metrics"]
        JUDGE["OllamaJudge<br/>DeepEvalBaseLLM wrapper"]
    end

    subgraph OLL["Ollama (local, 127.0.0.1:11434)"]
        LP["llama3 — product"]
        LJ["llama3 — judge"]
    end

    DATA["dataset/goldens.json"]
    OUT["results/*.json · *.csv · *.jsonl"]
    HTML["results/dashboard.html"]

    DATA --> RUN
    CFG --> RUN
    RUN --> PROD
    PROD -->|"AI_SOURCE=ollama"| LP
    PROD -.->|"AI_SOURCE=api"| API["Your product API"]
    RUN -->|spawn child process| EVAL
    EVAL --> JUDGE --> LJ
    EVAL --> RUN
    RUN --> DIAG --> OUT
    OUT --> DASH --> HTML
```

---

## 2. Runtime sequence (one evaluation)

```mermaid
sequenceDiagram
    autonumber
    participant U as User (npm run eval)
    participant R as runEval.js
    participant P as aiProduct.js
    participant O as Ollama (product)
    participant E as evaluate.py
    participant J as Ollama (judge)
    participant D as diagnostics.js

    U->>R: start run
    loop each golden case
        R->>P: callAIProduct(input)
        P->>O: /api/chat (system + user)
        O-->>P: actual_output (+ tokens)
        P-->>R: output + latency/tokens/retries
    end
    R->>R: write .tmp/augmented.json
    R->>E: spawn (input, actual, expected, context)
    loop each case × metric
        E->>J: structured generate (statements/verdicts/reason)
        J-->>E: JSON verdicts
    end
    E-->>R: .tmp/metrics.json (scores + reasons)
    R->>D: merge metrics + operational + labels
    D-->>R: matrices + root cause + flat records
    R->>R: write results/*.{json,csv,jsonl} + diagnostics.json
    R-->>U: console report
```

---

## 3. Data / artifact flow

```mermaid
flowchart LR
    G["goldens.json<br/>input · expected · context<br/>category/subcategory/journey"]
    A[".tmp/augmented.json<br/>+ actual_output"]
    M[".tmp/metrics.json<br/>per-case metric scores"]
    RJ["results.RUN.json<br/>summary + detail"]
    CSV["results.RUN.csv<br/>case × metric rows"]
    JL["results.flat.RUN.jsonl<br/>one object per case"]
    DG["diagnostics.RUN.json<br/>matrices + stage + worst"]
    H["dashboard.html<br/>trends across runs"]

    G --> A --> M --> RJ
    M --> CSV
    M --> JL
    M --> DG
    RJ --> H
    CSV --> H
    JL --> H
    DG --> H
```

---

## 4. Metric taxonomy

```mermaid
flowchart TB
    subgraph Response
        R1[Relevance]
        R2[Faithfulness]
        R3[Hallucination]
        R4["Correctness (GEval)"]
        R5["Completeness (GEval)"]
    end
    subgraph Retrieval
        C1[Context Precision]
        C2[Context Recall]
    end
    subgraph Safety
        S1[Toxicity]
        S2[Bias]
        S3[PII Leakage]
        S4["Jailbreak Resistance (GEval)"]
        S5["Data-Exfiltration Safety (GEval)"]
        S6["Sensitive-Info Safety (GEval)"]
    end
    subgraph Operational["Operational (JS harness, not DeepEval)"]
        O1[Latency]
        O2[Token usage]
        O3[Cost / query]
        O4[Failure rate]
        O5[Retry rate]
    end
```

---

## 5. Diagnostic drill‑down

```mermaid
flowchart TB
    M0["Matrix 0<br/>all metrics × category"] --> M1["Matrix 1<br/>metric × category + fail rate"]
    M1 --> M2["Matrix 2<br/>category × subcategory"]
    M2 --> M3["Matrix 3<br/>subcategory × journey"]
    M3 --> WC["Worst cases<br/>failStage + likely root cause"]

    subgraph Stages["Fail-stage attribution"]
        direction LR
        RET["Retrieval<br/>recall/precision"]
        RES["Response<br/>faithfulness/correctness/completeness/relevance"]
        SAF["Safety<br/>guardrail metrics"]
    end
    WC --> Stages
```
