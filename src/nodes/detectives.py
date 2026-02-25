"""
Detective Layer — LangGraph nodes that collect forensic evidence.

These nodes do NOT opine or score. They only produce structured Evidence objects
backed by concrete file paths, AST findings, and commit data.

Uses LangChain's ChatAnthropic with .with_structured_output() bound to the
Evidence Pydantic schema — ensuring structured output, not freeform text.
"""

import json
import os
from typing import Any, Dict, List

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from src.state import AgentState, Evidence
from src.tools.doc_tools import ingest_pdf, query_pdf
from src.tools.repo_tools import full_repo_analysis

# ---------------------------------------------------------------------------
# Shared LangChain Claude client with structured output
# ---------------------------------------------------------------------------

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-4-6")

# This LLM is bound to return Evidence objects — not freeform text
_evidence_llm = ChatAnthropic(model=CLAUDE_MODEL).with_structured_output(Evidence)


def _get_evidence(system: str, user: str) -> Evidence:
    """
    Call Claude via LangChain with .with_structured_output(Evidence).
    Retries once on failure before returning a fallback Evidence object.
    """
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    try:
        return _evidence_llm.invoke(messages)
    except Exception as first_err:
        print(f"[Detective] Structured output attempt 1 failed: {first_err}. Retrying...")
        try:
            return _evidence_llm.invoke(messages)
        except Exception as second_err:
            print(f"[Detective] Structured output attempt 2 failed: {second_err}. Using fallback.")
            return Evidence(
                goal="Evidence collection",
                found=False,
                content=None,
                location="llm_error",
                rationale=f"LLM failed after 2 attempts: {second_err}",
                confidence=0.0,
            )


# ---------------------------------------------------------------------------
# RepoInvestigator Node
# ---------------------------------------------------------------------------

REPO_SYSTEM_PROMPT = (
    "You are a forensic code investigator. Your job is to interpret technical "
    "evidence from a repository analysis and produce a structured Evidence object. "
    "Be precise, cite file paths and commit hashes. Do NOT invent findings. "
    "Set 'found' to true ONLY if the success pattern is clearly and directly met by the data."
)


def repo_investigator_node(state: AgentState) -> Dict:
    """
    Clones the target repository, runs all forensic checks, and produces
    structured Evidence objects for each rubric dimension targeting github_repo.
    Uses .with_structured_output(Evidence) for guaranteed Pydantic validation.
    """
    repo_url: str = state["repo_url"]
    rubric_dimensions: List[Dict] = state.get("rubric_dimensions", [])

    print(f"[RepoInvestigator] Cloning and analysing: {repo_url}")

    analysis = full_repo_analysis(repo_url)

    if "clone_error" in analysis:
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
    key_snippets = analysis.get("key_snippets", {})

    summary = {
        "commit_count": git_history.get("commit_count", 0),
        "commit_pattern": git_history.get("pattern", "unknown"),
        "commits": git_history.get("commits", [])[:10],
        "files": files[:60],
        "state_graph_found": graph_analysis.get("state_graph_found", False),
        "parallel_fan_out": graph_analysis.get("parallel_fan_out", False),
        "evidence_aggregator_node": graph_analysis.get("evidence_aggregator_node", False),
        "pydantic_basemodel": graph_analysis.get("pydantic_basemodel", False),
        "typeddict_used": graph_analysis.get("typeddict_used", False),
        "operator_reducers": graph_analysis.get("operator_reducers", False),
        "tempfile_sandboxing": graph_analysis.get("tempfile_sandboxing", False),
        "structured_output_enforcement": graph_analysis.get("structured_output_enforcement", False),
        "add_edge_calls": graph_analysis.get("add_edge_calls", [])[:20],
        "key_snippets": {k: v[:500] for k, v in key_snippets.items()},
        "errors": graph_analysis.get("errors", []),
    }

    repo_dimensions = [
        d for d in rubric_dimensions if d.get("target_artifact") == "github_repo"
    ]

    evidences: Dict[str, List[Evidence]] = {}

    for dim in repo_dimensions:
        dim_id = dim["id"]
        dim_name = dim["name"]

        user_msg = (
            f"Dimension: {dim_name}\n"
            f"Forensic Instruction: {dim.get('forensic_instruction', '')}\n\n"
            f"Repository Analysis Data:\n{json.dumps(summary, indent=2)}\n\n"
            f"Success pattern: {dim.get('success_pattern', 'N/A')}\n"
            f"Failure pattern: {dim.get('failure_pattern', 'N/A')}\n\n"
            "Produce an Evidence object based strictly on the data above. "
            "The 'goal' field should be the dimension name. "
            "The 'location' field should be the specific file path or git reference. "
            "The 'confidence' field should be between 0.0 and 1.0."
        )

        evidence = _get_evidence(REPO_SYSTEM_PROMPT, user_msg)
        evidences[dim_id] = [evidence]

    print(f"[RepoInvestigator] Generated evidence for {len(evidences)} dimensions")
    return {"evidences": evidences}


# ---------------------------------------------------------------------------
# DocAnalyst Node
# ---------------------------------------------------------------------------

DOC_SYSTEM_PROMPT = (
    "You are a forensic document analyst. Your job is to interpret a PDF report "
    "and produce a structured Evidence object. Be precise, cite exact passages. "
    "Do NOT invent content. Set 'found' to true only if the evidence clearly meets "
    "the success pattern."
)


def doc_analyst_node(state: AgentState) -> Dict:
    """
    Reads the PDF report, checks for conceptual depth, cross-references file paths.
    Uses .with_structured_output(Evidence) for guaranteed Pydantic validation.
    """
    pdf_path: str = state.get("pdf_path", "")
    rubric_dimensions: List[Dict] = state.get("rubric_dimensions", [])

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

    doc_dimensions = [
        d for d in rubric_dimensions
        if d.get("target_artifact") in ("pdf_report", "pdf_images")
    ]

    evidences: Dict[str, List[Evidence]] = {}

    for dim in doc_dimensions:
        dim_id = dim["id"]
        dim_name = dim["name"]
        forensic_instruction = dim.get("forensic_instruction", "")

        relevant_chunks = query_pdf(full_text, forensic_instruction, top_k=3)
        context = "\n---\n".join(relevant_chunks)

        summary = {
            "concept_depth": {
                k: {"depth": v["depth"], "found": v["found"]}
                for k, v in concept_depth.items()
            },
            "file_paths_mentioned": file_paths_mentioned,
            "relevant_excerpts": context[:2500],
        }

        user_msg = (
            f"Dimension: {dim_name}\n"
            f"Forensic Instruction: {forensic_instruction}\n\n"
            f"Document Analysis:\n{json.dumps(summary, indent=2)}\n\n"
            f"Success pattern: {dim.get('success_pattern', 'N/A')}\n"
            f"Failure pattern: {dim.get('failure_pattern', 'N/A')}\n\n"
            "Produce an Evidence object. The 'goal' should be the dimension name. "
            "The 'location' should reference the PDF section or passage. "
            "The 'confidence' field should be between 0.0 and 1.0."
        )

        evidence = _get_evidence(DOC_SYSTEM_PROMPT, user_msg)
        evidences[dim_id] = [evidence]

    print(f"[DocAnalyst] Generated evidence for {len(evidences)} dimensions")
    return {"evidences": evidences}


# ---------------------------------------------------------------------------
# Evidence Aggregator Node (Fan-In synchronisation point)
# ---------------------------------------------------------------------------

def evidence_aggregator_node(state: AgentState) -> Dict:
    """
    Fan-in synchronisation node. By the time LangGraph invokes this node,
    both repo_investigator and doc_analyst have completed and their evidences
    have been merged into state via the operator.ior reducer.

    Validates completeness and logs any missing dimensions.
    """
    evidences = state.get("evidences", {})
    total = sum(len(v) for v in evidences.values())
    print(f"[EvidenceAggregator] Collected {total} evidence items across {len(evidences)} dimensions")

    rubric_dimensions = state.get("rubric_dimensions", [])
    covered_ids = set(evidences.keys())
    all_ids = {d["id"] for d in rubric_dimensions}
    missing = all_ids - covered_ids
    if missing:
        print(f"[EvidenceAggregator] WARNING: Missing evidence for: {missing}")

    return {}
