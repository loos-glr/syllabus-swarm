"""syllabus-swarm — deterministic evaluation layer (System One / Jev).

This package hosts the **non-generative** decision use-cases that replaced
LLM-based control flow:

* :mod:`src.evaluators.modality_router` — routes a module to a generator.
* :mod:`src.evaluators.qa_scorer` — scores generated content against a rubric.
* :mod:`src.evaluators.syllabus_gate` — gates syllabus handoff to content generation.

Every evaluator accepts an injected :class:`src.system_one.SystemOneClient`, so
callers can supply a real client, a degraded offline double, or a fully
scripted fake in tests.
"""

from src.evaluators.modality_router import (
    COMPLEXITY_LEVELS,
    MODALITY_CRITERIA,
    ROUTING_MAP,
    ModalityRouting,
    RoutingState,
    get_modality_router,
    route_module,
    routing_target,
)
from src.evaluators.qa_scorer import (
    LAB_CRITERIA,
    THEORY_CRITERIA,
    QARubric,
    QAScoring,
    default_qa_rubric,
    get_qa_scorer,
    qa_needs_review,
    qa_passed,
    render_qa_report,
    rubric_from_profile,
    score_artifact,
    score_artifacts,
)
from src.evaluators.syllabus_gate import (
    QUALITY_LEVELS,
    REQUIRED_SECTIONS,
    SyllabusGate,
    assess_syllabus,
    check_required_sections,
    extract_duration_mentions,
    get_syllabus_gate,
    render_gate_report,
)

__all__ = [
    # modality_router
    "COMPLEXITY_LEVELS",
    "MODALITY_CRITERIA",
    "ROUTING_MAP",
    "ModalityRouting",
    "RoutingState",
    "get_modality_router",
    "route_module",
    "routing_target",
    # qa_scorer
    "LAB_CRITERIA",
    "THEORY_CRITERIA",
    "QARubric",
    "QAScoring",
    "default_qa_rubric",
    "get_qa_scorer",
    "qa_needs_review",
    "qa_passed",
    "render_qa_report",
    "rubric_from_profile",
    "score_artifact",
    "score_artifacts",
    # syllabus_gate
    "QUALITY_LEVELS",
    "REQUIRED_SECTIONS",
    "SyllabusGate",
    "assess_syllabus",
    "check_required_sections",
    "extract_duration_mentions",
    "get_syllabus_gate",
    "render_gate_report",
]
