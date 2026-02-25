"""
Judicial Layer — Three adversarial judge personas that debate evidence.

Each judge runs in parallel via LangGraph fan-out, analyzing the same
evidence through a completely different philosophical lens.

All judges use .with_structured_output(JudicialOpinion) — guaranteed
Pydantic validation, no freeform text parsing.
"""

import os
from typing import Dict, List

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from src.state import AgentState, Evidence, JudicialOpinion

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-4-6")

# ---------------------------------------------------------------------------
# Three distinct LLM instances — same model, radically different personas
# ---------------------------------------------------------------------------

_prosecutor_llm = ChatAnthropic(model=CLAUDE_MODEL).with_structured_output(JudicialOpinion)
_defense_llm    = ChatAnthropic(model=CLAUDE_MODEL).with_structured_output(JudicialOpinion)
_techlead_llm   = ChatAnthropic(model=CLAUDE_MODEL).with_structured_output(JudicialOpinion)

# ---------------------------------------------------------------------------
# Persona System Prompts — must be distinctly adversarial
# ---------------------------------------------------------------------------

PROSECUTOR_SYSTEM = """You are the Prosecutor in a Digital Courtroom. Your philosophy: "Trust No One. Assume Vibe Coding."

Your job is to scrutinize evidence for gaps, security flaws, and laziness. You are adversarial by design.

Rules you must follow:
- If parallelism is missing → charge "Orchestration Fraud" → score 1
- If judges return freeform text without Pydantic → charge "Hallucination Liability" → max score 2
- If os.system() is used without sandboxing → charge "Security Negligence" → score 1
- If state uses plain dicts instead of Pydantic → charge "Technical Debt" → max score 2
- Always look for what is MISSING, not what is present
- Be harsh but cite specific evidence. Never give benefit of the doubt.
- Your score should reflect the worst confirmed violation found.

You must return a JudicialOpinion with: judge="Prosecutor", a score 1-5, a harsh argument, and cited_evidence listing specific file paths or findings."""

DEFENSE_SYSTEM = """You are the Defense Attorney in a Digital Courtroom. Your philosophy: "Reward Effort and Intent. Look for the Spirit of the Law."

Your job is to highlight creative workarounds, deep thought, and genuine engineering effort — even when implementation is imperfect.

Rules you must follow:
- If the graph fails but AST parsing logic is sophisticated → argue "Deep Code Comprehension" → boost score from 1 to 3
- If commits show iterative struggle and progression → argue "Engineering Process" → reward higher score
- If ChiefJustice uses LLM synthesis but judge personas are highly distinct → argue "Role Separation Achieved" → score 3-4
- Look for evidence of understanding even when syntax is broken
- Emphasize what IS present and working, not what is missing
- Always find something genuinely praiseworthy before criticizing
- Your score should reflect the best reasonable interpretation of the evidence.

You must return a JudicialOpinion with: judge="Defense", a generous score 1-5, an optimistic argument, and cited_evidence listing specific strengths found."""

TECHLEAD_SYSTEM = """You are the Tech Lead in a Digital Courtroom. Your philosophy: "Does it actually work? Is it maintainable?"

Your job is to evaluate architectural soundness, code cleanliness, and practical viability. You are the pragmatic tie-breaker.

Rules you must follow:
- Ignore "vibe" and "struggle" — focus only on the artifacts
- If operator.add/ior reducers are present → parallelism is safe → acknowledge this explicitly
- If tempfile sandboxing is used → security standard met → acknowledge this
- If .with_structured_output() is used → structured output enforced → acknowledge this
- If architecture is modular and workable → this carries the HIGHEST weight for Architecture criterion
- Score 1 = broken and unmaintainable, 3 = works but has technical debt, 5 = production-grade
- You are the tie-breaker between Prosecutor (harsh) and Defense (generous)
- Provide specific remediation advice in your argument.

You must return a JudicialOpinion with: judge="TechLead", a realistic score 1-5, a pragmatic argument with remediation steps, and cited_evidence."""

# ---------------------------------------------------------------------------
# Shared retry wrapper
# ---------------------------------------------------------------------------

def _invoke_with_retry(llm, messages, judge_name: str, criterion_id: str) -> JudicialOpinion:
    """Invoke LLM with structured output, retry once on failure."""
    try:
        return llm.invoke(messages)
    except Exception as first_err:
        print(f"[{judge_name}] Attempt 1 failed: {first_err}. Retrying...")
        try:
            return llm.invoke(messages)
        except Exception as second_err:
            print(f"[{judge_name}] Attempt 2 failed: {second_err}. Using fallback.")
            return JudicialOpinion(
                judge=judge_name,
                criterion_id=criterion_id,
                score=1,
                argument=f"Judge failed to produce structured output after 2 attempts: {second_err}",
                cited_evidence=["llm_error"],
            )

# ---------------------------------------------------------------------------
# Helper: format evidence for a dimension
# ---------------------------------------------------------------------------

def _format_evidence(evidences: Dict, dimension_id: str) -> str:
    ev_list: List[Evidence] = evidences.get(dimension_id, [])
    if not ev_list:
        return "No evidence collected for this dimension."
    lines = []
    for ev in ev_list:
        status = "FOUND" if ev.found else "NOT FOUND"
        lines.append(
            f"- [{status}] {ev.goal}\n"
            f"  Location: {ev.location}\n"
            f"  Rationale: {ev.rationale}\n"
            f"  Confidence: {ev.confidence:.2f}"
        )
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Judge Nodes
# ---------------------------------------------------------------------------

def prosecutor_node(state: AgentState) -> Dict:
    """Prosecutor: adversarial lens, looks for violations and gaps."""
    evidences = state.get("evidences", {})
    rubric_dimensions = state.get("rubric_dimensions", [])
    opinions = []

    print(f"[Prosecutor] Analysing {len(rubric_dimensions)} dimensions...")

    for dim in rubric_dimensions:
        dim_id = dim["id"]
        evidence_text = _format_evidence(evidences, dim_id)

        user_msg = (
            f"Dimension: {dim['name']} (id: {dim_id})\n"
            f"Judicial Logic: {dim.get('forensic_instruction', '')}\n\n"
            f"Evidence collected by Detectives:\n{evidence_text}\n\n"
            f"Success pattern: {dim.get('success_pattern', '')}\n"
            f"Failure pattern: {dim.get('failure_pattern', '')}\n\n"
            "As the Prosecutor, render your harshest honest verdict. "
            "Your criterion_id must be exactly: " + dim_id
        )

        messages = [SystemMessage(content=PROSECUTOR_SYSTEM), HumanMessage(content=user_msg)]
        opinion = _invoke_with_retry(_prosecutor_llm, messages, "Prosecutor", dim_id)
        opinion.criterion_id = dim_id  # ensure correct id
        opinions.append(opinion)

    print(f"[Prosecutor] Rendered {len(opinions)} opinions")
    return {"opinions": opinions}


def defense_node(state: AgentState) -> Dict:
    """Defense Attorney: generous lens, rewards effort and intent."""
    evidences = state.get("evidences", {})
    rubric_dimensions = state.get("rubric_dimensions", [])
    opinions = []

    print(f"[Defense] Analysing {len(rubric_dimensions)} dimensions...")

    for dim in rubric_dimensions:
        dim_id = dim["id"]
        evidence_text = _format_evidence(evidences, dim_id)

        user_msg = (
            f"Dimension: {dim['name']} (id: {dim_id})\n"
            f"Judicial Logic: {dim.get('forensic_instruction', '')}\n\n"
            f"Evidence collected by Detectives:\n{evidence_text}\n\n"
            f"Success pattern: {dim.get('success_pattern', '')}\n"
            f"Failure pattern: {dim.get('failure_pattern', '')}\n\n"
            "As the Defense Attorney, find the most generous honest interpretation. "
            "Your criterion_id must be exactly: " + dim_id
        )

        messages = [SystemMessage(content=DEFENSE_SYSTEM), HumanMessage(content=user_msg)]
        opinion = _invoke_with_retry(_defense_llm, messages, "Defense", dim_id)
        opinion.criterion_id = dim_id
        opinions.append(opinion)

    print(f"[Defense] Rendered {len(opinions)} opinions")
    return {"opinions": opinions}


def techlead_node(state: AgentState) -> Dict:
    """Tech Lead: pragmatic lens, architectural soundness and maintainability."""
    evidences = state.get("evidences", {})
    rubric_dimensions = state.get("rubric_dimensions", [])
    opinions = []

    print(f"[TechLead] Analysing {len(rubric_dimensions)} dimensions...")

    for dim in rubric_dimensions:
        dim_id = dim["id"]
        evidence_text = _format_evidence(evidences, dim_id)

        user_msg = (
            f"Dimension: {dim['name']} (id: {dim_id})\n"
            f"Judicial Logic: {dim.get('forensic_instruction', '')}\n\n"
            f"Evidence collected by Detectives:\n{evidence_text}\n\n"
            f"Success pattern: {dim.get('success_pattern', '')}\n"
            f"Failure pattern: {dim.get('failure_pattern', '')}\n\n"
            "As the Tech Lead, give a pragmatic assessment focused on whether "
            "this actually works and is maintainable in production. "
            "Your criterion_id must be exactly: " + dim_id
        )

        messages = [SystemMessage(content=TECHLEAD_SYSTEM), HumanMessage(content=user_msg)]
        opinion = _invoke_with_retry(_techlead_llm, messages, "TechLead", dim_id)
        opinion.criterion_id = dim_id
        opinions.append(opinion)

    print(f"[TechLead] Rendered {len(opinions)} opinions")
    return {"opinions": opinions}
