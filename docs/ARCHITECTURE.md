## 1. Bounded Contexts
The application is logically segmented into the following architectural domains:
*   **Intake & Profiling Context:** Ingesting configuration parameters and school defaults (e.g., `school_defaults.yaml`, `beroeps2_profile.yaml`).
*   **Curriculum Orchestration Context:** Defining the high-level educational structure, chaining modules, and orchestrating the educational flow.
*   **Content Generation Context:** Authoring specific educational materials, both theoretical and practical.
*   **Quality Assurance & Validation Context:** Reviewing generated syllabi and theory against rubrics or standards to ensure pedagogical integrity.
*   **Export & Persistence Context:** Structuring the final outputs into manifests and writing them to the file system.

## 2. Ubiquitous Language
*   **Swarm/Crew:** A coordinated group of role-playing AI agents executing sequential or hierarchical tasks.
*   **Agent:** A discrete, specialized persona powered by an LLM.
*   **Task:** A definitive unit of work assigned to an Agent.
*   **Profile:** A YAML-based configuration defining the constraints and requirements for a specific educational module.
*   **Manifest:** The final structural metadata describing the exported curriculum artifacts.
*   **HITL (Human-in-the-Loop):** A cyclical state allowing human review and feedback to regenerate agent outputs.

## 3. Technology Stack Manifest
*   **Runtime:** Python 3.12
*   **Testing:** `pytest`, `pytest-mock` (Core testing framework for TDD)
*   **Data Validation:** `pydantic`
*   **Configuration:** `pyyaml`, `python-dotenv`
*   **Agentic Orchestration:** `crewai` (or equivalent multi-agent state machine)

## 4. Clean Architecture Topological Mapping
*   **Layer 1: Enterprise Business Rules (Domain Entities):** `src/models.py`, `config/`. Defines Pydantic data models. Dependencies: None.
*   **Layer 2: Application Business Rules (Use Cases):** `src/tasks/`, `src/crews/`. Orchestrates the swarm, handles cyclic HITL state machine logic. Dependencies: Layer 1.
*   **Layer 3: Interface Adapters (Gateways & Presenters):** `src/agents/`, `src/exporters/`. Defines personas and translates swarm memory into deployable artifact structures. Dependencies: Layer 2.
*   **Layer 4: Frameworks & Drivers (Infrastructure):** `src/main.py`, `src/llm_factory.py`. Bootstraps CLI, handles IO, and connects to LLM providers using standard SDKs. Dependencies: Layer 3.

## 5. Modality Routing & Video-as-Code (VaC) Extension
*   **Domain Expansion (Layer 1):** `ModalityType` Enum (`CLASSIC_READER`, `INTERACTIVE_WEB`, `INTERACTIVE_CLI`, `VIDEO_AS_CODE`), `ModalityDecision` (with System One provenance: `confidence`, `probabilities`, `model_version`), and `RemotionManifest`.
*   **Decision Layer (retirement):** The `media_strategist` agent has been **retired**. Modality routing is now a deterministic System One decision produced by `src/evaluators/modality_router.py`. Routing is exhaustive over `ModalityType` (`ROUTING_MAP`) and mapped onto the state machine by `MODALITY_STATE_MAP` / `next_state_after_routing()` in `src/crews/syllabus_crew.py`.
*   **Agent Expansion (Layer 3):** `video_engineer` specializes in generating deterministic temporal code (React/Remotion JSX) instead of prose.
*   **Polyglot Export (Layer 3):** The `file_writer` intercepts `RemotionManifest` data to persist valid `.tsx`/`.jsx` files into a dedicated `src/export/vac/` directory, completely isolated from Markdown generation.

## 6. System One (Jev) Decision Layer
*   **Purpose:** Offload routing, QA scoring and syllabus gating from generative LLMs to a fast, schema-driven, **non-generative** System One model (TypeSafe AI's `Jev`). Generative models keep handling all natural-language content creation.
*   **Infrastructure (Layer 4):** `src/system_one.py` (client protocol + `TypeSafeSystemOneClient` adapter + offline/test `FakeSystemOneClient`) and `build_system_one_client()` in `src/llm_factory.py` (3-tier env chain: `SYSTEM_ONE_{TASK}_{PROPERTY}` → `SYSTEM_ONE_{PROPERTY}` → hardcoded `jev-latest`).
*   **Domain (Layer 1):** Typed decision schemas in `src/models.py`: `RubricCriterion`, `QAScore`, `SyllabusGateDecision`.
*   **Use cases (Layer 2/3):** `src/evaluators/modality_router.py`, `src/evaluators/qa_scorer.py`, `src/evaluators/syllabus_gate.py`.
*   **Three primitives:** `choice` (modality routing), `score` (QA rubric criteria, syllabus quality), `noul` (QA verdict, syllabus completeness/coherence).
*   **Layer split:** exact rules (syntax, required sections, duration extraction) stay in plain Python; the model only makes bounded semantic judgments. Arithmetic is never delegated to the model.
*   **Escalation:** every decision carries a `confidence`; scores below the configured threshold set `needs_review=True` so the orchestrator escalates instead of acting on an un-calibrated guess.
*   **Degradation:** `SYSTEM_ONE_ENABLED=0` selects a heuristic offline client whose answers are always un-confident, so the pipeline never auto-approves a decision it cannot calibrate.