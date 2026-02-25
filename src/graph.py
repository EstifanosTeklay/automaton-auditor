"""
StateGraph definition for the Automaton Auditor (Interim Submission).

Architecture for the interim:
    START
      |
      v
  [context_builder]          <- loads rubric.json into state
      |
     / \\
    /   \\
[repo_investigator] [doc_analyst]   <- parallel fan-out (Detectives)
    \\   /
     \\ /
      v
  [evidence_aggregator]      <- fan-in synchronisation
      |
      v
     END

Judges and ChiefJustice are NOT wired yet (final submission).
"""

import json
import os
from pathlib import Path
from typing import Any, Dict

from langgraph.graph import END, START, StateGraph

from src.nodes.detectives import (
    doc_analyst_node,
    evidence_aggregator_node,
    repo_investigator_node,
)
from src.state import AgentState

# ---------------------------------------------------------------------------
# Rubric Loader
# ---------------------------------------------------------------------------

_RUBRIC_PATH = Path(__file__).parent.parent / "rubric.json"


def load_rubric() -> list:
    """Load rubric dimensions from rubric.json."""
    if not _RUBRIC_PATH.exists():
        raise FileNotFoundError(f"rubric.json not found at {_RUBRIC_PATH}")
    with open(_RUBRIC_PATH) as f:
        data = json.load(f)
    return data["dimensions"]


# ---------------------------------------------------------------------------
# Context Builder Node
# ---------------------------------------------------------------------------

def context_builder_node(state: AgentState) -> Dict:
    """
    Loads the rubric and injects it into state before the Detectives run.
    This is the entry point of the graph.
    """
    dimensions = load_rubric()
    print(f"[ContextBuilder] Loaded {len(dimensions)} rubric dimensions")
    return {
        "rubric_dimensions": dimensions,
        "evidences": {},
        "opinions": [],
        "final_report": None,
    }


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph StateGraph.

    Fan-out: context_builder -> repo_investigator AND doc_analyst (parallel)
    Fan-in:  both detectives -> evidence_aggregator
    """
    builder = StateGraph(AgentState)

    # --- Register nodes ---
    builder.add_node("context_builder", context_builder_node)
    builder.add_node("repo_investigator", repo_investigator_node)
    builder.add_node("doc_analyst", doc_analyst_node)
    builder.add_node("evidence_aggregator", evidence_aggregator_node)

    # --- Wire edges ---

    # Entry point
    builder.add_edge(START, "context_builder")

    # Fan-out: both detectives run in parallel after context is built
    builder.add_edge("context_builder", "repo_investigator")
    builder.add_edge("context_builder", "doc_analyst")

    # Fan-in: both detectives must complete before aggregation
    builder.add_edge("repo_investigator", "evidence_aggregator")
    builder.add_edge("doc_analyst", "evidence_aggregator")

    # End (Judges will be inserted here in the final submission)
    builder.add_edge("evidence_aggregator", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def run_detective_swarm(repo_url: str, pdf_path: str) -> Dict[str, Any]:
    """
    Run the detective layer against a repository URL and PDF report.
    Returns the final AgentState after evidence aggregation.
    """
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
    print(f"AUTOMATON AUDITOR — Detective Swarm")
    print(f"Target Repo : {repo_url}")
    print(f"PDF Report  : {pdf_path}")
    print(f"{'='*60}\n")

    final_state = graph.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m src.graph <repo_url> <pdf_path>")
        sys.exit(1)

    result = run_detective_swarm(sys.argv[1], sys.argv[2])

    print("\n--- EVIDENCE COLLECTED ---")
    for dim_id, ev_list in result.get("evidences", {}).items():
        for ev in ev_list:
            status = "✅" if ev.found else "❌"
            print(f"  {status} [{dim_id}] {ev.goal} (confidence: {ev.confidence:.2f})")
            print(f"     Location : {ev.location}")
            print(f"     Rationale: {ev.rationale[:120]}...")
