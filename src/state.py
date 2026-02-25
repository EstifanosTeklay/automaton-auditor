"""
State definitions for the Automaton Auditor.
Uses Pydantic BaseModel and TypedDict with reducers to safely
handle parallel agent execution without data overwriting.
"""

import operator
from typing import Annotated, Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Detective Output
# ---------------------------------------------------------------------------

class Evidence(BaseModel):
    """Structured forensic evidence produced by a Detective agent."""

    goal: str = Field(description="The forensic goal this evidence addresses")
    found: bool = Field(description="Whether the artifact or pattern was found")
    content: Optional[str] = Field(
        default=None,
        description="The actual content extracted (code snippet, commit list, etc.)"
    )
    location: str = Field(
        description="File path, commit hash, or section reference where evidence was found"
    )
    rationale: str = Field(
        description="The detective's reasoning for its confidence level"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score between 0.0 and 1.0"
    )


# ---------------------------------------------------------------------------
# Judge Output
# ---------------------------------------------------------------------------

class JudicialOpinion(BaseModel):
    """A single judge's verdict on one rubric criterion."""

    judge: Literal["Prosecutor", "Defense", "TechLead"] = Field(
        description="Which judge persona produced this opinion"
    )
    criterion_id: str = Field(
        description="The rubric dimension ID this opinion addresses"
    )
    score: int = Field(
        ge=1, le=5,
        description="Score from 1 (worst) to 5 (best)"
    )
    argument: str = Field(
        description="Full reasoning for the assigned score"
    )
    cited_evidence: List[str] = Field(
        default_factory=list,
        description="List of evidence goal IDs or location references cited"
    )


# ---------------------------------------------------------------------------
# Chief Justice Output
# ---------------------------------------------------------------------------

class CriterionResult(BaseModel):
    """The Chief Justice's final ruling on a single rubric criterion."""

    dimension_id: str
    dimension_name: str
    final_score: int = Field(ge=1, le=5)
    judge_opinions: List[JudicialOpinion]
    dissent_summary: Optional[str] = Field(
        default=None,
        description="Required when score variance across judges exceeds 2"
    )
    remediation: str = Field(
        description="Specific, file-level instructions for improvement"
    )


class AuditReport(BaseModel):
    """The complete audit report produced by the Chief Justice."""

    repo_url: str
    executive_summary: str
    overall_score: float
    criteria: List[CriterionResult]
    remediation_plan: str


# ---------------------------------------------------------------------------
# Graph State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """
    The shared state passed between all nodes in the LangGraph StateGraph.

    Reducers:
    - evidences uses operator.ior (dict merge) so parallel Detectives can
      each write their own key without overwriting siblings.
    - opinions uses operator.add (list append) so parallel Judges can each
      append their JudicialOpinion without overwriting siblings.
    """

    repo_url: str
    pdf_path: str

    # Rubric loaded from rubric.json — list of dimension dicts
    rubric_dimensions: List[Dict]

    # Parallel-safe evidence accumulator: { dimension_id: [Evidence, ...] }
    evidences: Annotated[Dict[str, List[Evidence]], operator.ior]

    # Parallel-safe opinion accumulator
    opinions: Annotated[List[JudicialOpinion], operator.add]

    # Final output
    final_report: Optional[AuditReport]
