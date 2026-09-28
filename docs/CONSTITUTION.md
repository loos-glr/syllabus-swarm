## 1. Strict Mandates
*   **Mandate 1: Strict TDD (Red-Green-Refactor).** No application logic shall be written until a failing `pytest` is committed. Tests must assert domain logic isolated from infrastructure.
*   **Mandate 2: Model Agnosticism via Configuration.** Zero hardcoded model parameters or provider URLs. The `llm_factory.py` MUST rely purely on environment variables (e.g., `BASE_URL`, `API_KEY`) to establish client connections using standard provider SDKs. 
*   **Mandate 3: Clean Architecture Boundaries.** Inner layers (Entities, Use Cases) must never import from outer layers (Adapters, Infrastructure). Data crossing boundaries must be strictly serialized via `pydantic`.
*   **Mandate 4: Type Strictness.** Python 3.12 type hinting is mandatory. `mypy` strict mode passes are required for all modules.

## 2. Product Spec (System Flows)
*   **Flow 1: Ingestion & Validation:** User executes `main.py` passing a configuration Profile. Infrastructure reads YAML, passing it to Layer 1 `pydantic` models for strict validation. Result: Strongly-typed `CourseGraph` entity.
*   **Flow 2: Swarm Execution (Orchestration):** `CourseGraph` is passed to the `syllabus_crew`. The Curriculum Architect designs the module flow. The Theory Instructor and Lab Developer generate content concurrently. The QA Reviewer validates outputs.
*   **Flow 3: HITL Cyclic Routing:** After QA Review, the swarm pauses. The CLI prompts for human feedback. If approved, proceed to Export. If rejected/critiqued, route feedback back to specific generating agents for a new iteration.
*   **Flow 4: Artifact Export:** `file_writer` maps the in-memory syllabus to the `Manifest` schema and persists it to the output directory as Markdown and JSON artifacts.

## 3. Polyglot Generation Spec (Modality Routing Flows)
*   **Flow 5: Modality Strategy (deterministic):** For each curriculum `Module`, the System One model (`src/evaluators/modality_router.py`) evaluates pedagogical complexity with a `choice` + `score` round-trip and outputs a validated, typed `ModalityDecision` carrying `confidence` and per-option `probabilities`. The former `media_strategist` agent has been retired — no LLM judges the route.
*   **Flow 6: Divergent Generation:** The Swarm state machine reads the `ModalityDecision` via `MODALITY_STATE_MAP` / `next_state_after_routing()`. If `VIDEO_AS_CODE`, execution routes to the `video_engineer` (state `VIDEO_GENERATING`). Otherwise it stays in the standard `GENERATING` flow handled by the `theory_instructor`.
*   **Flow 7: Deterministic VaC Output:** The `video_engineer` generates a structural `RemotionManifest` (React/JSX and composition data). It strictly avoids generating narrative prose or pixel-based media.
*   **Flow 8: Polyglot Export:** The `file_writer` dynamically handles `RemotionManifest` entities, creating physically distinct `.tsx` artifacts on the file system.

## 4. Deterministic Decision Spec (System One / Jev)
*   **Flow 9: Decision Layer:** Control-flow judgments (routing, QA verdicts, syllabus gating) are produced by a non-generative System One model returning typed `choice` / `score` / `noul` answers. Generative LLMs remain responsible for *content creation* only.
*   **Flow 10: QA Verdict:** `src/evaluators/qa_scorer.py` scores each artifact against a typed rubric derived from the cohort profile and returns a `QAScore`; the `verdict` field (not an LLM's prose) is the authoritative pass/fail signal.
*   **Flow 11: Syllabus Gate:** `src/evaluators/syllabus_gate.py` combines plain-Python structural checks with a `noul` completeness gate to return a `SyllabusGateDecision`; the orchestrator advances only when `complete` is True.
*   **Flow 12: Confidence Escalation:** Every decision exposes `confidence` and `needs_review`. Decisions below the configured threshold are escalated rather than silently applied; `SYSTEM_ONE_ENABLED=0` forces un-calibrated offline answers that can never auto-approve.