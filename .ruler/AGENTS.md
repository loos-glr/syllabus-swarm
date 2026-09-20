# AGENTS.md

Centralised AI agent instructions for syllabus-swarm — a Dutch MBO4 vocational
education curriculum generator built with CrewAI.

---

## Project Context

- **Project:** syllabus-swarm
- **Purpose:** Automated generation of Dutch MBO4 (vocational education)
  curricula using a multi-agent CrewAI swarm.
- **Language:** Python 3.12
- **Framework:** CrewAI with OpenRouter as the LLM gateway
- **Output:** Syllabi, lab assignments, and rubrics in Markdown / structured
  formats

---

## LLM Factory Pattern — CRITICAL

Agents **MUST NOT** read environment variables or instantiate LLMs directly
(e.g. `ChatOpenAI(...)`, `LLM(...)`, `os.getenv("OPENROUTER_API_KEY")`).

Instead, every agent **MUST** obtain its LLM through the canonical factory:

```python
from src.llm_factory import build_llm_for_agent, CURRICULUM_ARCHITECT

llm = build_llm_for_agent(CURRICULUM_ARCHITECT)
agent = Agent(role="...", goal="...", llm=llm, ...)
```

Available role constants:
`CURRICULUM_ARCHITECT`, `LAB_DEVELOPER`, `OUTPUT_EXPORTER`,
`INTAKE_SPECIALIST`, `QA_REVIEWER`, `THEORY_INSTRUCTOR`,
`EDUCATION_DIRECTOR`, `MEDIA_STRATEGIST`, `VIDEO_ENGINEER`.

The factory handles the 3-tier fallback chain (per-agent override →
agent-wide default → hardcoded sensible default) and always
targets `https://openrouter.ai/api/v1`.

When adding a new agent, register its role constant in `src/llm_factory.py`
and use `build_llm_for_agent(YOUR_ROLE)` — never hardcode model IDs.

---

## Disk I/O Constraint — CRITICAL

Agents and tasks **MUST NOT** use raw `open()` for writing files. All file
writing **MUST** go through one of:

1. **`src.exporters.file_writer`** (the canonical module):
   - `write_file(path, content, *, force=False)` — write a single file with
     overwrite protection
   - `write_directory_tree(base_path, files_dict, *, force=False)` — batch
     writes
   - `write_syllabus(course_name, content)` — syllabus convenience helper
   - `write_rubric(course_name, content)` — rubric convenience helper
   - `write_lab_file(course_name, tier, variant, filename, content)` — lab
     file helper
   - `write_remotion_manifest(manifest, *, force=False)` — VaC .tsx export
     (Video-as-Code React/Remotion compositions)

2. **`OutputExportTool`** (when writing from within a CrewAI task/tool
   context).

These utilities ensure:
- Cross-platform safe paths (`pathlib`)
- Automatic parent-directory creation
- Overwrite blocked unless `force=True`
- Mandated output directory structure (`output/syllabus/`, `output/labs/`,
  `output/rubrics/`)
- Safe filename sanitisation

---

## CrewAI Delegation Scope — CRITICAL

Agents with `allow_delegation=True` (such as the **QA Reviewer** and
**Education Director**) **MUST** be instantiated in the **same**
`Crew(agents=[...])` array as their target co-workers.

This ensures the feedback loops (QA → architect, Director → all agents)
can actually resolve at runtime. Do **not** split delegating agents into a
separate crew — CrewAI delegation only traverses agents in the same crew
instance.

---

## Code Style

### Python Version
- Target **Python 3.12**. Use `from __future__ import annotations` in every
  module.

### Type Hinting
- Strict, complete type annotations on all public functions and methods.
- Use modern syntax: `list[str]`, `dict[str, int]`, `str | None` (not
  `Optional[str]`), `tuple[str, ...]`.

### Formatter & Linter
- **Ruff** for both formatting and linting.
- Configuration (in `pyproject.toml`):
  - `line-length = 100`
  - `quote-style = "double"`
  - `indent-style = "space"`
  - `docstring-code-format = true`
  - `target-version = "py312"`
- Run `ruff check . && ruff format .` before committing.

### Testing
- **pytest** via `pytest.ini` in the project root.
- Test files live in `tests/` mirroring the `src/` structure.
- Use descriptive test function names (`test_<function>_<scenario>`).
- All new features must include tests.

### Project Configuration
- `pyproject.toml` — build system, Ruff settings, project metadata
- `requirements.txt` — runtime dependencies
- `pytest.ini` — pytest defaults
- `.env.example` — template for environment variables (OpenRouter API key,
  per-agent model overrides)
