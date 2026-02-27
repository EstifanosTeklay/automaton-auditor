"""
Full StateGraph — Automaton Auditor (Final Submission).

Architecture:
    START → context_builder
              ├→ repo_investigator ─┐
              └→ doc_analyst        ├→ evidence_aggregator
                                   │        │
                              (conditional edge)
                                   │        │
                              ┌────┘        └────────────────────┐
                              ▼                                   ▼
                         handle_error              ┌─ prosecutor  │
                              │                    ├─ defense     │ (parallel)
                              │                    └─ techlead    │
                              │                         └─────────┤
                              │                                   ▼
                              │                          chief_justice
                              │                                   │
                              └───────────────────────────────────┘
                                                                  ▼
                                                                 END
"""

import json
from pathlib import Path
from typing import Any, Dict, Literal

from langgraph.graph import END, START, StateGraph

from src.nodes.detectives import (
    doc_analyst_node,
    evidence_aggregator_node,
    repo_investigator_node,
)
from src.nodes.judges import defense_node, prosecutor_node, techlead_node
from src.nodes.justice import chief_justice_node
from src.state import AgentState

_RUBRIC_PATH = Path(__file__).parent.parent / "rubric.json"


def load_rubric() -> list:
    if not _RUBRIC_PATH.exists():
        raise FileNotFoundError(f"rubric.json not found at {_RUBRIC_PATH}")
    with open(_RUBRIC_PATH) as f:
        return json.load(f)["dimensions"]


def context_builder_node(state: AgentState) -> Dict:
    dimensions = load_rubric()
    print(f"[ContextBuilder] Loaded {len(dimensions)} rubric dimensions")
    return {"rubric_dimensions": dimensions, "evidences": {}, "opinions": [], "final_report": None}


def handle_error_node(state: AgentState) -> Dict:
    evidences = state.get("evidences", {})
    for dim_id, ev_list in evidences.items():
        for ev in ev_list:
            if ev.confidence == 0.0:
                print(f"[ErrorHandler] {dim_id}: {ev.rationale}")
    return {}


def route_after_aggregation(state: AgentState) -> Literal["handle_error", "judges"]:
    """
    Conditional edge: only hard-fails on a clone error.
    Missing PDF dimensions are soft failures — judges still run with partial evidence.
    """
    evidences = state.get("evidences", {})

    # Only stop if the repo itself could not be cloned
    if "clone_failure" in evidences:
        print("[Router] Clone failure → handle_error")
        return "handle_error"

    # Everything else proceeds to judges
    total = sum(len(v) for v in evidences.values())
    print(f"[Router] {total} evidence items collected → proceeding to judges")
    return "judges"


def judicial_fanout_node(state: AgentState) -> Dict:
    """
    Pass-through node that triggers the 3 parallel judges simultaneously.
    """
    evidences = state.get("evidences", {})
    print(f"[JudicialFanout] Triggering 3 parallel judges on "
          f"{len(evidences)} evidence items")
    return {}


def build_graph() -> StateGraph:
    """
    Complete StateGraph with two fan-out/fan-in pairs:
      1. Detectives:  context_builder → [repo_investigator || doc_analyst] → evidence_aggregator
      2. Judges:      judicial_fanout → [prosecutor || defense || techlead] → chief_justice
    Plus a conditional error-routing edge after evidence_aggregator.
    """
    builder = StateGraph(AgentState)

    # Register all nodes
    builder.add_node("context_builder",     context_builder_node)
    builder.add_node("repo_investigator",   repo_investigator_node)
    builder.add_node("doc_analyst",         doc_analyst_node)
    builder.add_node("evidence_aggregator", evidence_aggregator_node)
    builder.add_node("handle_error",        handle_error_node)
    builder.add_node("judges",              judicial_fanout_node)
    builder.add_node("prosecutor",          prosecutor_node)
    builder.add_node("defense",             defense_node)
    builder.add_node("techlead",            techlead_node)
    builder.add_node("chief_justice",       chief_justice_node)

    # Entry
    builder.add_edge(START, "context_builder")

    # Detective Fan-Out
    builder.add_edge("context_builder", "repo_investigator")
    builder.add_edge("context_builder", "doc_analyst")

    # Detective Fan-In
    builder.add_edge("repo_investigator", "evidence_aggregator")
    builder.add_edge("doc_analyst",       "evidence_aggregator")

    # Conditional Edge: clone failure → error, everything else → judges
    builder.add_conditional_edges(
        "evidence_aggregator",
        route_after_aggregation,
        {
            "handle_error": "handle_error",
            "judges":       "judges",
        },
    )

    # Error path terminates
    builder.add_edge("handle_error", END)

    # Judicial Fan-Out (3-way parallel)
    builder.add_edge("judges", "prosecutor")
    builder.add_edge("judges", "defense")
    builder.add_edge("judges", "techlead")

    # Judicial Fan-In
    builder.add_edge("prosecutor", "chief_justice")
    builder.add_edge("defense",    "chief_justice")
    builder.add_edge("techlead",   "chief_justice")

    # Final output
    builder.add_edge("chief_justice", END)

    return builder.compile()


def run_auditor(repo_url: str, pdf_path: str) -> Dict[str, Any]:
    graph = build_graph()
    initial_state: AgentState = {
        "repo_url": repo_url,
        "pdf_path": pdf_path,
        "rubric_dimensions": [],
        "evidences": {},
        "opinions": [],
        "final_report": None,
    }
    print(f"\n{'='*60}")
    print(f"AUTOMATON AUDITOR — Full Swarm")
    print(f"Target Repo : {repo_url}")
    print(f"PDF Report  : {pdf_path}")
    print(f"{'='*60}\n")
    return graph.invoke(initial_state)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m src.graph <repo_url> <pdf_path>")
        sys.exit(1)
    result = run_auditor(sys.argv[1], sys.argv[2])
    report = result.get("final_report")
    if report:
        print(f"\nOverall Score: {report.overall_score}/5")
