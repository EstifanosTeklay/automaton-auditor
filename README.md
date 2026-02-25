# Automaton Auditor 🔍⚖️

**Week 2 — FDE Challenge | Interim Submission**

An automated code auditing swarm built with LangGraph and Claude. The system forensically analyses a GitHub repository and a PDF architectural report, producing structured evidence that will feed into a three-judge dialectical system in the final submission.

---

## Architecture Overview

```
                         ┌─────────────────────┐
                         │   context_builder   │  ← loads rubric.json
                         └─────────┬───────────┘
                                   │ Fan-Out
                    ┌──────────────┴──────────────┐
                    ▼                              ▼
        ┌─────────────────────┐      ┌─────────────────────┐
        │  repo_investigator  │      │    doc_analyst      │
        │  (Code Detective)   │      │ (Document Detective)│
        │                     │      │                     │
        │ • git clone (sandboxed)   │ • PDF ingestion      │
        │ • git log analysis  │      │ • RAG-lite query    │
        │ • AST parsing       │      │ • concept depth     │
        │ • file structure    │      │ • path cross-ref    │
        └──────────┬──────────┘      └──────────┬──────────┘
                   │                             │
                   └──────────────┬──────────────┘
                                  │ Fan-In
                         ┌────────▼────────┐
                         │ evidence_aggregator │
                         └────────┬────────┘
                                  │
                   ┌──────────────┴──────────────┐
                   │    [Judges — Final Sub]      │
                   │  Prosecutor || Defense ||    │
                   │       TechLead (parallel)   │
                   └──────────────┬──────────────┘
                                  │
                         ┌────────▼────────┐
                         │  ChiefJustice   │
                         │  [Final Sub]    │
                         └─────────────────┘
```

### Key Design Decisions

| Decision | Why |
|---|---|
| **Pydantic BaseModel** over plain dicts | Enforces strict typing, prevents silent data corruption, gives free validation |
| **operator.ior / operator.add reducers** | Allows parallel Detective nodes to write to the same state key without race conditions |
| **tempfile.TemporaryDirectory()** | Cloned code never lands in the live working directory — prevents path traversal and repo pollution |
| **subprocess.run() over os.system()** | Captures stdout/stderr, checks return codes, prevents shell injection |
| **AST parsing over regex** | Regex is brittle against formatting changes; AST understands Python structure semantically |
| **Claude API directly** | Anthropic's native SDK gives full control over structured JSON output via prompting |

---

## Project Structure

```
automaton-auditor/
├── src/
│   ├── state.py                  # Pydantic state definitions + AgentState
│   ├── graph.py                  # LangGraph StateGraph (detective layer)
│   ├── tools/
│   │   ├── repo_tools.py         # Sandboxed git clone, git log, AST analysis
│   │   └── doc_tools.py          # PDF ingestion, RAG-lite query, concept depth
│   └── nodes/
│       └── detectives.py         # RepoInvestigator + DocAnalyst + EvidenceAggregator
├── rubric.json                   # Machine-readable rubric (agent's Constitution)
├── pyproject.toml                # uv-managed dependencies
├── .env.example                  # Required environment variables (no secrets)
├── reports/
│   └── interim_report.pdf        # Architecture decisions PDF
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- An Anthropic API key

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install Dependencies

```bash
# Clone this repo
git clone <your-repo-url>
cd automaton-auditor

# Install all dependencies with uv
uv sync

# Or with pip (fallback)
pip install -e .
```

### Configure Environment

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

---

## Running the Detective Swarm

```bash
# Basic usage
uv run python -m src.graph <github_repo_url> <path_to_pdf>

# Example
uv run python -m src.graph \
  https://github.com/someuser/week2-submission \
  ./reports/submission_report.pdf
```

### Example Output

```
============================================================
AUTOMATON AUDITOR — Detective Swarm
Target Repo : https://github.com/someuser/week2-submission
PDF Report  : ./reports/submission_report.pdf
============================================================

[ContextBuilder] Loaded 10 rubric dimensions
[RepoInvestigator] Cloning and analysing: https://github.com/...
[DocAnalyst] Ingesting PDF: ./reports/submission_report.pdf
[EvidenceAggregator] Collected 18 evidence items across 9 dimensions

--- EVIDENCE COLLECTED ---
  ✅ [git_forensic_analysis] Git Forensic Analysis (confidence: 0.92)
     Location : git log
     Rationale: 7 commits found with clear progression pattern...
  ✅ [state_management_rigor] State Management Rigor (confidence: 0.95)
     Location : src/state.py
     Rationale: Pydantic BaseModel and TypedDict found with reducers...
  ❌ [graph_orchestration] Graph Orchestration Architecture (confidence: 0.85)
     Location : src/graph.py
     Rationale: Linear flow detected, no parallel fan-out found...
```

---

## Enable LangSmith Tracing

Add to your `.env`:
```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=automaton-auditor
```

Traces will appear at [smith.langchain.com](https://smith.langchain.com).

---

## What's Coming in the Final Submission

- `src/nodes/judges.py` — Prosecutor, Defense Attorney, Tech Lead running in parallel
- `src/nodes/justice.py` — ChiefJustice with deterministic conflict resolution rules
- Full graph wiring: Detectives → Judges → ChiefJustice
- Markdown audit report output
- Audit reports on self and peer repositories

---

## Dependency Management with uv (Reproducible Installs)

This project uses `uv` with a committed `uv.lock` file for fully reproducible installs. Every dependency is pinned to an exact version with a hash — meaning `uv sync` gives every developer and CI environment the identical package versions.

```bash
# Install exact locked versions (recommended — fully reproducible)
uv sync

# Update lockfile after changing pyproject.toml
uv lock

# Add a new dependency and update lockfile atomically
uv add some-package
```

Why this matters: `pip install -r requirements.txt` with `>=` version constraints can silently install different versions on different machines, causing subtle bugs. The `uv.lock` file eliminates this entirely.
