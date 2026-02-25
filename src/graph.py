"""
StateGraph definition for the Automaton Auditor (Interim Submission).

Architecture:
    START
      |
      v
  [context_builder]
      |
     / \
    /   \
[repo_investigator] [doc_analyst]   <- parallel fan-out
    \   /
     \ /
      v
  [evidence_aggregator]
      |
      v (conditional edge: check for clone/pdf errors)
  [handle_error] OR continues
      |
      v
     END

Judges and ChiefJustice will be wired in the final submission.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Literal

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
    """Loads the rubric and initialises state before Detectives run."""
    dimensions = load_rubric()
    print(f"[ContextBuilder] Loaded {len(dimensions)} rubric dimensions")
    return {
        "rubric_dimensions": dimensions,
        "evidences": {},
        "opinions": [],
        "final_report": None,
    }


# ---------------------------------------------------------------------------
# Error Handler Node
# ---------------------------------------------------------------------------

def handle_error_node(state: AgentState) -> Dict:
    """
    Invoked when a critical detective failure is detected (e.g. clone failed,
    PDF not found). Logs the error and terminates gracefully rather than
    crashing the graph.
    """
    evidences = state.get("evidences", {})
    errors = []
    for dim_id, ev_list in evidences.items():
        for ev in ev_list:
            if not ev.found and ev.location in ("parse_error", "llm_error"):
                errors.append(f"{dim_id}: {ev.rationale}")

    print(f"[ErrorHandler] Critical failures detected:\n" + "\n".join(errors))
    return {}


# ---------------------------------------------------------------------------
# Conditional Edge: route after aggregation
# ---------------------------------------------------------------------------

def route_after_aggregation(
    state: AgentState,
) -> Literal["handle_error", "__end__"]:
    """
    Conditional edge function. Inspects the aggregated evidence for
    critical failures (clone error, PDF parse error) and routes to the
    error handler if found — otherwise proceeds to END (or Judges in
    the final submission).

    This prevents the Judicial layer from running on empty/corrupt evidence.
    """
    evidences = state.get("evidences", {})

    # Check for hard clone failure
    if "clone_failure" in evidences:
        print("[Router] Clone failure detected — routing to error handler")
        return "handle_error"

    if "pdf_failure" in evidences:
        print("[Router] PDF failure detected — routing to error handler")
        return "handle_error"

    # Check if too many detectives returned zero-confidence evidence
    low_confidence_count = 0
    for ev_list in evidences.values():
        for ev in ev_list:
            if ev.confidence == 0.0:
                low_confidence_count += 1

    if low_confidence_count >= 3:
        print(f"[Router] {low_confidence_count} zero-confidence findings — routing to error handler")
        return "handle_error"

    print("[Router] Evidence looks good — proceeding")
    return "__end__"


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph StateGraph.

    Fan-out  : context_builder -> repo_investigator AND doc_analyst (parallel)
    Fan-in   : both detectives -> evidence_aggregator
    Conditional edge: evidence_aggregator -> handle_error OR END
    """
    builder = StateGraph(AgentState)

    # --- Register nodes ---
    builder.add_node("context_builder", context_builder_node)
    builder.add_node("repo_investigator", repo_investigator_node)
    builder.add_node("doc_analyst", doc_analyst_node)
    builder.add_node("evidence_aggregator", evidence_aggregator_node)
    builder.add_node("handle_error", handle_error_node)

    # --- Wire edges ---

    # Entry point
    builder.add_edge(START, "context_builder")

    # Fan-out: both detectives run in parallel
    builder.add_edge("context_builder", "repo_investigator")
    builder.add_edge("context_builder", "doc_analyst")

    # Fan-in: both detectives must complete before aggregation
    builder.add_edge("repo_investigator", "evidence_aggregator")
    builder.add_edge("doc_analyst", "evidence_aggregator")

    # Conditional edge: route based on evidence health
    builder.add_conditional_edges(
        "evidence_aggregator",
        route_after_aggregation,
        {
            "handle_error": "handle_error",
            "__end__": END,
        },
    )

    # Error handler always terminates
    builder.add_edge("handle_error", END)

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
            print(f"     Rationale: {ev.rationale[:120]}")
