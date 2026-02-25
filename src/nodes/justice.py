"""
Chief Justice Synthesis Engine.

This node does NOT use an LLM to average scores. It applies hardcoded
deterministic Python rules to resolve judicial conflicts and produce
a final, defensible verdict for each rubric criterion.

The three rules (hardcoded, not prompted):
  1. Rule of Security    — confirmed vulnerabilities cap score at 3
  2. Rule of Evidence    — Detective facts override Judge opinions
  3. Rule of Functionality — Tech Lead confirmation carries highest weight
     for architecture criteria
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from src.state import AgentState, AuditReport, CriterionResult, JudicialOpinion

# ---------------------------------------------------------------------------
# Security violation keywords the Prosecutor might flag
# ---------------------------------------------------------------------------

SECURITY_VIOLATION_KEYWORDS = [
    "os.system", "shell injection", "unsanitized", "security negligence",
    "command injection", "arbitrary code", "security flaw", "vulnerability",
]

HALLUCINATION_KEYWORDS = [
    "hallucination", "fabricated", "does not exist", "not found",
    "missing file", "file not present",
]


# ---------------------------------------------------------------------------
# Deterministic Conflict Resolution Rules
# ---------------------------------------------------------------------------

def _rule_of_security(
    prosecutor_opinion: JudicialOpinion,
    current_score: int,
) -> int:
    """
    Rule of Security: if the Prosecutor identifies a confirmed security
    vulnerability, cap the final score at 3 regardless of Defense arguments.
    """
    argument_lower = prosecutor_opinion.argument.lower()
    if any(kw in argument_lower for kw in SECURITY_VIOLATION_KEYWORDS):
        if prosecutor_opinion.score <= 2:  # Prosecutor is genuinely alarmed
            capped = min(current_score, 3)
            if capped < current_score:
                print(f"    [Rule of Security] Score capped {current_score} → {capped}")
            return capped
    return current_score


def _rule_of_evidence(
    defense_opinion: JudicialOpinion,
    evidences: Dict,
    criterion_id: str,
    current_score: int,
) -> int:
    """
    Rule of Evidence: if the Defense claims something positive but Detective
    evidence shows the artifact is missing or not found, overrule the Defense.
    """
    ev_list = evidences.get(criterion_id, [])
    if not ev_list:
        return current_score

    # If all detective evidence says NOT FOUND but Defense is generous
    all_not_found = all(not ev.found for ev in ev_list)
    defense_is_generous = defense_opinion.score >= 4

    if all_not_found and defense_is_generous:
        overruled_score = min(current_score, 2)
        print(f"    [Rule of Evidence] Defense overruled — no evidence found. "
              f"Score {current_score} → {overruled_score}")
        return overruled_score

    return current_score


def _rule_of_functionality(
    techlead_opinion: JudicialOpinion,
    criterion_id: str,
    scores: Dict[str, int],
) -> int:
    """
    Rule of Functionality: for architecture-related criteria, if the Tech Lead
    confirms the architecture is modular and workable, their score carries
    the highest weight (not a simple average).
    """
    architecture_criteria = {
        "graph_orchestration", "state_management_rigor",
        "safe_tool_engineering", "structured_output_enforcement",
    }

    if criterion_id in architecture_criteria:
        tl_score = techlead_opinion.score
        avg = sum(scores.values()) / len(scores)

        # Weight Tech Lead score at 50%, others at 25% each
        weighted = round((tl_score * 0.5) + (avg * 0.5))
        print(f"    [Rule of Functionality] Architecture criterion — "
              f"TechLead weighted score applied: {weighted}")
        return weighted

    return round(sum(scores.values()) / len(scores))


def _requires_dissent(scores: Dict[str, int]) -> bool:
    """Returns True if score variance across judges exceeds 2."""
    values = list(scores.values())
    return (max(values) - min(values)) > 2


def _build_dissent_summary(
    opinions: List[JudicialOpinion],
    final_score: int,
) -> str:
    """Build a concise dissent summary explaining the conflict."""
    lines = []
    for op in opinions:
        lines.append(f"- {op.judge} scored {op.score}/5: {op.argument[:150]}...")
    lines.append(f"\nChief Justice final ruling: {final_score}/5")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-Criterion Resolution
# ---------------------------------------------------------------------------

def _resolve_criterion(
    criterion_id: str,
    criterion_name: str,
    opinions_for_criterion: List[JudicialOpinion],
    evidences: Dict,
) -> CriterionResult:
    """
    Apply the three deterministic rules to resolve a single criterion.
    """
    # Index opinions by judge
    by_judge: Dict[str, JudicialOpinion] = {}
    for op in opinions_for_criterion:
        by_judge[op.judge] = op

    prosecutor = by_judge.get("Prosecutor")
    defense    = by_judge.get("Defense")
    techlead   = by_judge.get("TechLead")

    # Fallback if a judge is missing (shouldn't happen but be defensive)
    scores = {
        "Prosecutor": prosecutor.score if prosecutor else 3,
        "Defense":    defense.score    if defense    else 3,
        "TechLead":   techlead.score   if techlead   else 3,
    }

    # Start with Tech Lead weighted average (Rule of Functionality)
    if techlead:
        final_score = _rule_of_functionality(techlead, criterion_id, scores)
    else:
        final_score = round(sum(scores.values()) / len(scores))

    # Apply Rule of Security (can only lower the score)
    if prosecutor:
        final_score = _rule_of_security(prosecutor, final_score)

    # Apply Rule of Evidence (can only lower the score)
    if defense:
        final_score = _rule_of_evidence(defense, evidences, criterion_id, final_score)

    # Clamp to 1-5
    final_score = max(1, min(5, final_score))

    # Dissent summary required when variance > 2
    dissent = None
    if _requires_dissent(scores):
        dissent = _build_dissent_summary(opinions_for_criterion, final_score)
        print(f"    [Dissent Required] Variance={max(scores.values()) - min(scores.values())} "
              f"for {criterion_id}")

    # Remediation: use Tech Lead's argument (most actionable)
    remediation = (
        techlead.argument if techlead
        else "No Tech Lead opinion available — review manually."
    )

    return CriterionResult(
        dimension_id=criterion_id,
        dimension_name=criterion_name,
        final_score=final_score,
        judge_opinions=opinions_for_criterion,
        dissent_summary=dissent,
        remediation=remediation,
    )


# ---------------------------------------------------------------------------
# Chief Justice Node
# ---------------------------------------------------------------------------

def chief_justice_node(state: AgentState) -> Dict:
    """
    Synthesises all judicial opinions into a final AuditReport using
    deterministic Python rules — NOT another LLM prompt.

    Outputs a structured AuditReport and writes it as a Markdown file.
    """
    opinions: List[JudicialOpinion] = state.get("opinions", [])
    evidences = state.get("evidences", {})
    rubric_dimensions = state.get("rubric_dimensions", [])
    repo_url = state.get("repo_url", "unknown")

    print(f"\n[ChiefJustice] Synthesising {len(opinions)} opinions across "
          f"{len(rubric_dimensions)} dimensions...")

    # Group opinions by criterion_id
    by_criterion: Dict[str, List[JudicialOpinion]] = defaultdict(list)
    for op in opinions:
        by_criterion[op.criterion_id].append(op)

    # Build a name lookup from rubric
    name_lookup = {d["id"]: d["name"] for d in rubric_dimensions}

    # Resolve each criterion
    criteria_results: List[CriterionResult] = []
    for criterion_id, criterion_opinions in by_criterion.items():
        print(f"  Resolving: {criterion_id}")
        result = _resolve_criterion(
            criterion_id=criterion_id,
            criterion_name=name_lookup.get(criterion_id, criterion_id),
            opinions_for_criterion=criterion_opinions,
            evidences=evidences,
        )
        criteria_results.append(result)

    # Overall score = mean of all criterion final scores
    if criteria_results:
        overall = round(
            sum(r.final_score for r in criteria_results) / len(criteria_results), 2
        )
    else:
        overall = 0.0

    # Executive summary
    executive_summary = (
        f"Audit of {repo_url} completed. "
        f"Overall score: {overall}/5 across {len(criteria_results)} criteria. "
        f"The Chief Justice applied three deterministic rules: Rule of Security "
        f"(vulnerability cap), Rule of Evidence (fact supremacy), and Rule of "
        f"Functionality (Tech Lead weighting for architecture criteria)."
    )

    # Remediation plan: collect all Tech Lead advice
    remediation_items = []
    for r in criteria_results:
        if r.final_score < 4:
            remediation_items.append(
                f"### {r.dimension_name} (Score: {r.final_score}/5)\n{r.remediation}"
            )
    remediation_plan = "\n\n".join(remediation_items) if remediation_items else "No critical remediations required."

    report = AuditReport(
        repo_url=repo_url,
        executive_summary=executive_summary,
        overall_score=overall,
        criteria=criteria_results,
        remediation_plan=remediation_plan,
    )

    # Serialise to Markdown
    _write_markdown_report(report)

    print(f"[ChiefJustice] Final verdict: {overall}/5")
    return {"final_report": report}


# ---------------------------------------------------------------------------
# Markdown Report Writer
# ---------------------------------------------------------------------------

def _write_markdown_report(report: AuditReport) -> str:
    """Serialise AuditReport to a Markdown file in audit/report_onself_generated/."""
    output_dir = Path("audit/report_onself_generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "audit_report.md"

    lines = [
        "# Automaton Auditor — Audit Report",
        "",
        "## Executive Summary",
        "",
        report.executive_summary,
        "",
        f"**Overall Score: {report.overall_score}/5**",
        "",
        "---",
        "",
        "## Criterion Breakdown",
        "",
    ]

    for cr in report.criteria:
        lines += [
            f"### {cr.dimension_name}",
            f"**Final Score: {cr.final_score}/5**",
            "",
        ]

        for op in cr.judge_opinions:
            lines += [
                f"#### {op.judge} — Score: {op.score}/5",
                op.argument,
                "",
                f"*Cited evidence: {', '.join(op.cited_evidence) if op.cited_evidence else 'none'}*",
                "",
            ]

        if cr.dissent_summary:
            lines += [
                "#### ⚖️ Dissent Summary",
                cr.dissent_summary,
                "",
            ]

        lines += [
            "#### 🔧 Remediation",
            cr.remediation,
            "",
            "---",
            "",
        ]

    lines += [
        "## Remediation Plan",
        "",
        report.remediation_plan,
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ChiefJustice] Report written to {output_path}")
    return str(output_path)
