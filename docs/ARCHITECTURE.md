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