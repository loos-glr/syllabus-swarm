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