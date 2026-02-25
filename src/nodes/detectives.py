"""
Detective Layer — LangGraph nodes that collect forensic evidence.

These nodes do NOT opine or score. They only produce structured Evidence objects
backed by concrete file paths, AST findings, and commit data.

Each node reads from AgentState and writes back a keyed evidence dict entry
that gets merged via the operator.ior reducer.
"""

import json
import os
from typing import Any, Dict, List

from anthropic import Anthropic

from src.state import AgentState, Evidence
from src.tools.doc_tools import (
    analyze_concept_depth,
    extract_file_paths_from_text,
    ingest_pdf,
    query_pdf,
)
from src.tools.repo_tools import full_repo_analysis

# ---------------------------------------------------------------------------
# Shared Anthropic client
# ---------------------------------------------------------------------------

_client = Anthropic()  # reads ANTHROPIC_API_KEY from env automatically

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-4-6")


def _ask_claude(system: str, user: str, max_tokens: int = 1500) -> str:
    """Helper: single Claude call returning the text response."""
    response = _client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text


# ---------------------------------------------------------------------------
# RepoInvestigator Node
# ---------------------------------------------------------------------------

def repo_investigator_node(state: AgentState) -> Dict:
    """
    Clones the target repository, runs all forensic checks, and produces
    structured Evidence objects for each rubric dimension that targets
    the github_repo artifact.
    """
    repo_url: str = state["repo_url"]
    rubric_dimensions: List[Dict] = state.get("rubric_dimensions", [])

    print(f"[RepoInvestigator] Cloning and analysing: {repo_url}")

    # Run all forensic tools
    analysis = full_repo_analysis(repo_url)

    if "clone_error" in analysis:
        # Return a single failure evidence for all repo dimensions
        failure_evidence = Evidence(
            goal="Repository Clone",
            found=False,
            content=None,
            location=repo_url,
            rationale=f"Clone failed: {analysis['clone_error']}",
            confidence=1.0,
        )
        return {"evidences": {"clone_failure": [failure_evidence]}}

    git_history = analysis.get("git_history", {})
    graph_analysis = analysis.get("graph_analysis", {})
    files = analysis.get("files", [])

    # Build a compact JSON summary to pass to Claude for nuanced assessment
    summary = {
        "commit_count": git_history.get("commit_count", 0),
        "commit_pattern": git_history.get("pattern", "unknown"),
        "commits": git_history.get("commits", [])[:10],  # first 10 commits
        "files": files[:60],  # cap to keep context manageable
        "state_graph_found": graph_analysis.get("state_graph_found", False),
        "parallel_fan_out": graph_analysis.get("parallel_fan_out", False),
        "evidence_aggregator_node": graph_analysis.get("evidence_aggregator_node", False),
        "pydantic_basemodel": graph_analysis.get("pydantic_basemodel", False),
        "typeddict_used": graph_analysis.get("typeddict_used", False),
        "operator_reducers": graph_analysis.get("operator_reducers", False),
        "tempfile_sandboxing": graph_analysis.get("tempfile_sandboxing", False),
        "structured_output_enforcement": graph_analysis.get("structured_output_enforcement", False),
        "add_edge_calls": graph_analysis.get("add_edge_calls", [])[:20],
        "errors": graph_analysis.get("errors", []),
    }

    # --- Per-dimension evidence generation ---
    repo_dimensions = [
        d for d in rubric_dimensions if d.get("target_artifact") == "github_repo"
    ]

    evidences: Dict[str, List[Evidence]] = {}

    SYSTEM_PROMPT = (
        "You are a forensic code investigator. Your job is to interpret technical "
        "evidence from a repository analysis and produce a structured JSON Evidence object. "
        "Be precise, cite file paths and commit hashes, and do NOT invent findings. "
        "Respond ONLY with valid JSON matching this schema:\n"
        '{"goal": str, "found": bool, "content": str|null, "location": str, '
        '"rationale": str, "confidence": float}'
    )

    for dim in repo_dimensions:
        dim_id = dim["id"]
        dim_name = dim["name"]
        forensic_instruction = dim.get("forensic_instruction", "")

        user_msg = (
            f"Dimension: {dim_name}\n"
            f"Forensic Instruction: {forensic_instruction}\n\n"
            f"Repository Analysis Data:\n{json.dumps(summary, indent=2)}\n\n"
            "Based on this data, produce the Evidence JSON object for this dimension. "
            "Set 'found' to true only if the success pattern is clearly met. "
            f"Success pattern: {dim.get('success_pattern', 'N/A')}\n"
            f"Failure pattern: {dim.get('failure_pattern', 'N/A')}"
        )

        try:
            raw = _ask_claude(SYSTEM_PROMPT, user_msg, max_tokens=600)
            # Strip markdown code fences if present
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = json.loads(raw)
            evidence = Evidence(**data)
        except Exception as exc:
            evidence = Evidence(
                goal=dim_name,
                found=False,
                content=None,
                location="parse_error",
                rationale=f"Claude response parse failed: {exc}",
                confidence=0.0,
            )

        evidences[dim_id] = [evidence]

    print(f"[RepoInvestigator] Generated evidence for {len(evidences)} dimensions")
    return {"evidences": evidences}


# ---------------------------------------------------------------------------
# DocAnalyst Node
# ---------------------------------------------------------------------------

def doc_analyst_node(state: AgentState) -> Dict:
    """
    Reads the PDF report, checks for conceptual depth, extracts file path
    references, and cross-references them against repo evidence already
    in state (if available).
    """
    pdf_path: str = state.get("pdf_path", "")
    rubric_dimensions: List[Dict] = state.get("rubric_dimensions", [])
    repo_evidences: Dict = state.get("evidences", {})

    print(f"[DocAnalyst] Ingesting PDF: {pdf_path}")

    doc_data = ingest_pdf(pdf_path)

    if "error" in doc_data:
        failure_evidence = Evidence(
            goal="PDF Ingestion",
            found=False,
            content=None,
            location=pdf_path,
            rationale=f"PDF ingestion failed: {doc_data['error']}",
            confidence=1.0,
        )
        return {"evidences": {"pdf_failure": [failure_evidence]}}

    full_text: str = doc_data["full_text"]
    concept_depth: Dict = doc_data["concept_depth"]
    file_paths_mentioned: List[str] = doc_data["file_paths_mentioned"]

    # Cross-reference: which mentioned file paths actually exist in repo?
    repo_files: List[str] = []
    for evidence_list in repo_evidences.values():
        for ev in evidence_list:
            if hasattr(ev, "content") and ev.content:
                # The repo investigator stores files list in content sometimes
                pass
    # Simpler approach: check against the analysis if available
    # (DocAnalyst runs in parallel so repo evidence might not be available yet)
    verified_paths: List[str] = []
    hallucinated_paths: List[str] = []
    # We flag these for the Chief Justice to resolve post-aggregation
    # For now we store the raw mentions
    unverified_paths = file_paths_mentioned

    # --- Per-dimension evidence generation ---
    doc_dimensions = [
        d for d in rubric_dimensions
        if d.get("target_artifact") in ("pdf_report", "pdf_images")
    ]

    evidences: Dict[str, List[Evidence]] = {}

    SYSTEM_PROMPT = (
        "You are a forensic document analyst. Your job is to interpret a PDF report "
        "and produce a structured Evidence JSON object. Be precise, cite exact passages. "
        "Do NOT invent content. "
        "Respond ONLY with valid JSON matching this schema:\n"
        '{"goal": str, "found": bool, "content": str|null, "location": str, '
        '"rationale": str, "confidence": float}'
    )

    for dim in doc_dimensions:
        dim_id = dim["id"]
        dim_name = dim["name"]
        forensic_instruction = dim.get("forensic_instruction", "")

        # Get relevant chunks for this dimension
        relevant_chunks = query_pdf(full_text, forensic_instruction, top_k=3)
        context = "\n---\n".join(relevant_chunks)

        summary = {
            "concept_depth": {
                k: {"depth": v["depth"], "found": v["found"]}
                for k, v in concept_depth.items()
            },
            "file_paths_mentioned": unverified_paths,
            "relevant_excerpts": context[:3000],
        }

        user_msg = (
            f"Dimension: {dim_name}\n"
            f"Forensic Instruction: {forensic_instruction}\n\n"
            f"Document Analysis:\n{json.dumps(summary, indent=2)}\n\n"
            "Produce the Evidence JSON object. "
            f"Success pattern: {dim.get('success_pattern', 'N/A')}\n"
            f"Failure pattern: {dim.get('failure_pattern', 'N/A')}"
        )

        try:
            raw = _ask_claude(SYSTEM_PROMPT, user_msg, max_tokens=600)
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = json.loads(raw)
            evidence = Evidence(**data)
        except Exception as exc:
            evidence = Evidence(
                goal=dim_name,
                found=False,
                content=None,
                location="parse_error",
                rationale=f"Claude response parse failed: {exc}",
                confidence=0.0,
            )

        evidences[dim_id] = [evidence]

    print(f"[DocAnalyst] Generated evidence for {len(evidences)} dimensions")
    return {"evidences": evidences}


# ---------------------------------------------------------------------------
# Evidence Aggregator Node (Fan-In synchronisation point)
# ---------------------------------------------------------------------------

def evidence_aggregator_node(state: AgentState) -> Dict:
    """
    Synchronisation node. Waits until all parallel Detectives have finished
    and their evidences have been merged into state via the operator.ior reducer.

    In the interim submission, this node just logs and passes through.
    In the full submission it can validate completeness and flag missing evidence.
    """
    evidences = state.get("evidences", {})
    total = sum(len(v) for v in evidences.values())
    print(f"[EvidenceAggregator] Collected {total} evidence items across {len(evidences)} dimensions")

    # Validate: check for missing dimensions
    rubric_dimensions = state.get("rubric_dimensions", [])
    covered_ids = set(evidences.keys())
    all_ids = {d["id"] for d in rubric_dimensions}
    missing = all_ids - covered_ids
    if missing:
        print(f"[EvidenceAggregator] WARNING: Missing evidence for dimensions: {missing}")

    # Pass state through unchanged — aggregation happened via the reducer
    return {}
