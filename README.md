# syllabus-swarm 🐝

[![CI](https://github.com/MelvinLoos/syllabus-swarm/actions/workflows/ci.yml/badge.svg)](https://github.com/MelvinLoos/syllabus-swarm/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

> A local multi-agent workspace for generating vocational-level software development curriculum.

**syllabus-swarm** uses AI agents (powered by [CrewAI](https://www.crewai.com/) / [LangChain](https://www.langchain.com/)) connected to specialized language models via **OpenRouter** to collaboratively design, develop, and package complete course materials — syllabi, interactive theory artifacts, tiered coding labs, and evaluation rubrics.

---

## Quick Start

```bash
# 1. Clone the repository
git clone git@github.com:MelvinLoos/syllabus-swarm.git
cd syllabus-swarm

# 2. Set up the virtual environment (requires Python 3.12+)
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API keys
cp .env.example .env
# Edit .env and paste your OpenRouter API key

# 5. Verify the connection
python3.12 -m src.llm_factory
```

---

## Architecture

syllabus-swarm is built on six specialized AI agents, each assigned a model optimized for its specific role:

| Agent | Role | Default Model | Rationale |
|---|---|---|---|
| **Intake Specialist** | Interviews the user to extract technical and pedagogical requirements mapped to Dutch SBB Kwalificatiedossiers | `deepseek/deepseek-v4-pro` | Strong reasoning for synthesising rich course context from user answers, with deep knowledge of MBO4 vocational education pathways (BOL/BBL) and kerntaken (P1-K1 through P4-K1). |
| **Curriculum Architect** | Designs syllabi using the Humanics framework (data literacy, technological literacy, human literacy) + experiential learning | `deepseek/deepseek-v4-pro` | State-of-the-art multi-step reasoning for crafting logically coherent, pedagogically sound syllabi that span weeks of content across three integrated literacies. |
| **Theory Instructor** | Transforms abstract syllabus concepts into interactive learning artifacts (HTML/JS visualizations, pausing terminal scripts, Mermaid.js diagrams) | `deepseek/deepseek-v4-pro` | Strong writing + code generation for producing self-contained, runnable interactive artifacts that vocational students can engage with before starting hands-on labs. |
| **Lab & Project Developer** | Generates tiered hands-on coding exercises with starter code and fully-commented solution keys | `openrouter/qwen/qwen3-coder` | Purpose-built for programming tasks — produces cleaner starter code, more idiomatic solutions, and fewer hallucinated API calls than general-purpose models. |
| **QA Reviewer** | Reviews all generated labs for technical correctness (syntax, imports, runnability) and MBO4 didactic appropriateness | `openrouter/qwen/qwen3-coder` | Purpose-built for code understanding and review — catches syntax errors, missing imports, hallucinated variables, and didactic issues before they reach students. |
| **Output Exporter** | Compiles and packages all materials into clean directory structures and a consolidated manifest | `deepseek/deepseek-v4-flash-latest` | Low-latency, low-cost completions ideal for manifest generation, file assembly, and Markdown packaging — reliability without burning reasoning-token budgets. |

All models are served via **OpenRouter** (`https://openrouter.ai/api/v1`).

### Pydantic Models

| Model | Location | Purpose |
|---|---|---|
| `CourseSpecification` | `src/main.py` | Structured output from Intake Specialist: `course_context`, `primary_language`, plus optional static constraints (`grading_scale`, `student_pathway`, `year_level`, `hardware_constraints`) |
| `IntakeSession` | `src/main.py` | Serializable record of a completed intake interview (questions, answers, synthesised `CourseSpecification`) |
| `CourseGraph` | `src/models.py` | Machine-readable course metadata — **composes** `CourseSpecification` rather than duplicating fields |
| `ModuleSummary` | `src/models.py` | Lightweight per-module record (`title`, `duration_weeks`, `hours_per_week`, `topics`) |

### Output Structure

| Agent | Output |
|---|---|
| Intake Specialist | `output/<run_id>/intake_session.json` (auto-saved) |
| Curriculum Architect | `output/<run_id>/syllabus/` |
| Theory Instructor | `output/<run_id>/theory/` (interactive HTML/JS, terminal scripts, Mermaid.js diagrams) |
| Lab & Project Developer | `output/<run_id>/labs/` |
| QA Reviewer | `output/<run_id>/labs/` (review reports and delegated fixes) |
| Output Exporter | `output/<run_id>/README.md` (manifest), `output/<run_id>/course_graph.json` (machine-readable) |

---

## Model Configuration

Every agent obtains its LLM through a shared factory (`src/llm_factory.py`) that resolves model, temperature, top_p, and max_tokens through a **3-tier fallback chain**.

### Environment Variable Convention

```
AGENT_{ROLE}_{PROPERTY}
```

- **`{ROLE}`** — uppercase snake_case agent identifier: `CURRICULUM_ARCHITECT`, `LAB_DEVELOPER`, `OUTPUT_EXPORTER`, `INTAKE_SPECIALIST`, `QA_REVIEWER`, `THEORY_INSTRUCTOR`
- **`{PROPERTY}`** — `MODEL`, `TEMPERATURE`, `MAX_TOKENS`, or `TOP_P`

Supported properties and their defaults:

| Property | Default | Description |
|---|---|---|
| `MODEL` | `deepseek/deepseek-v4-pro` (fallback) | OpenRouter model identifier |
| `TEMPERATURE` | `0.2` | Generation randomness (0.0 = deterministic, 1.0 = creative) |
| `MAX_TOKENS` | `8192` | Maximum completion tokens per agent call |
| `TOP_P` | `0.1` | Nucleus sampling threshold |

### Fallback Chain (highest to lowest priority)

```
1. AGENT_{ROLE}_{PROPERTY}    ← Per-agent override (most specific)
          │
2. AGENT_DEFAULT_{PROPERTY}   ← Catch-all default for all agents
          │
3. Hardcoded defaults         ← Values in src/llm_factory.py
```

### Hardcoded Fallback

The hardcoded catch-all model is `deepseek/deepseek-v4-pro`. Any agent that lacks both a per-agent override and an `AGENT_DEFAULT_MODEL` will use this model.

### Per-Agent Model Defaults (v2)

These defaults are set in `.env.example` and take effect when you copy it to `.env`:

```bash
# Curriculum Architect — deep reasoning for syllabus design
AGENT_CURRICULUM_ARCHITECT_MODEL=deepseek/deepseek-v4-pro

# Theory Instructor — strong writing + code generation for interactive artifacts
AGENT_THEORY_INSTRUCTOR_MODEL=deepseek/deepseek-v4-pro

# Lab & Project Developer — code generation specialist
AGENT_LAB_DEVELOPER_MODEL=openrouter/qwen/qwen3-coder

# QA Reviewer — code review and didactic analysis
AGENT_QA_REVIEWER_MODEL=openrouter/qwen/qwen3-coder

# Output Exporter — fast, deterministic packaging
AGENT_OUTPUT_EXPORTER_MODEL=deepseek/deepseek-v4-flash-latest
```

To override any agent's model, set the corresponding environment variable before running:

```bash
AGENT_CURRICULUM_ARCHITECT_MODEL=anthropic/claude-sonnet-4 python src/main.py "Python Basics"
```

> See [OpenRouter Models](https://openrouter.ai/models) for a complete list of available model identifiers.

### Verifying Configuration

To see exactly which model each agent is using (with full environment variable resolution):

```bash
python -m src.llm_factory
```

This prints the resolved model, temperature, top_p, and max_tokens for every known agent.

---

## Usage

```bash
# Full pipeline: intake + syllabus + theory + labs + QA review + manifest
python src/main.py "Data Science with Python"

# Syllabus only (skip theory, lab generation, and QA review)
python src/main.py "Full-Stack Web Development" --skip-labs

# Load a cohort profile (pre-populates static constraints, skips intake questions)
python src/main.py "Laravel Web Development" --profile config/profiles/program1_profile.yaml

# Resume a previous run (skip intake, re-run agents)
python src/main.py "ML Basics" --resume-from output/2026-08-22_153000_ML_Basics

# Load a saved intake session (skip interactive interview)
python src/main.py "Advanced PHP" --load-session output/2026-08-22_153000_ML_Basics/intake_session.json

# Chain modules: inject prerequisites from a previous course
python src/main.py "Period 3 Project" --builds-upon 2026-08-22_153000_ML_Basics

# Interactive prompt
python src/main.py
```

---

## Configuration System

syllabus-swarm supports a layered configuration model that separates static institutional constraints from dynamic, per-course intake:

```
config/
├── school_defaults.yaml          # Immutable institution-wide defaults
└── profiles/
    ├── program1_profile.yaml     # Example: PHP/Laravel — BOL Pathway (Year 2)
    └── scripting2_profile.yaml   # Example: Python scripting — BBL Pathway
```

### Layering Model

```
1. school_defaults.yaml     ←  Institution-wide (MBO4 standards, O/V/G grading,
                                BYOD constraints, SBB kerntaken)
         │
2. cohort profile            ←  Per-track overrides (year level, tech stack,
         │                      student pathway, kerntaken emphasis)
         │
3. Intake Specialist         ←  Interactive interview fills remaining gaps
```

When a `--profile` is loaded, the Intake Specialist automatically skips questions for any field already populated (grading scale, student pathway, year level, hardware constraints).

---

## Curriculum Memory & Continuity Engine

The **Curriculum Memory & Continuity Engine** (Epic #7) transitions syllabus-swarm from a stateless script to a stateful curriculum platform.

### Intake Session Persistence

Every completed intake interview is automatically saved to `output/<run_id>/intake_session.json`. Instructors can reload a session with `--load-session` to clone and tweak a profile for the next module instead of starting from scratch.

### Module Chaining (`--builds-upon`)

When the `--builds-upon <previous_course_slug>` flag is used, the pipeline reads the previous course's output, extracts its **Learning Objectives** and **Key Concepts**, and injects them as prerequisites into the Curriculum Architect's context. This enables automatic carry-over of prior knowledge (e.g., "Use the database designed in Period 2") without manual prompting.

The resolver prefers the machine-readable `course_graph.json` (see below) when available, and falls back to parsing the markdown syllabus.

### Course Graph Export

Alongside the visual `README.md` manifest, the exporter generates a machine-readable `course_graph.json` containing structured metadata:

- **`CourseGraph`** — composes `CourseSpecification` (no field duplication), plus `course_slug`, `learning_objectives`, `key_concepts`, `prerequisites`, and ordered `modules`
- **`ModuleSummary`** — lightweight per-module record with `title`, `duration_weeks`, `hours_per_week`, and `topics`

Both models are defined in `src/models.py` and are the canonical source of structured course metadata for downstream tooling (module chaining, LMS import, curriculum analytics).

---

## Troubleshooting

### "cannot import name 'UTC' from 'datetime'" / "No module named 'crewai'"

This means an **older Python (3.9) is being used**. The project requires Python 3.12+.

- Make sure the project virtualenv is active: `source .venv/bin/activate`.
- If `python --version` still reports an older version *after* activating, the
  venv's `python3` symlink may point at a system interpreter. Recreate it:

  ```bash
  rm -f .venv/bin/python .venv/bin/python3 .venv/bin/python3.9 .venv/bin/pip3.9
  python3.12 -m venv .venv
  source .venv/bin/activate
  ```

- Or invoke the correct interpreter directly: `python3.12 src/main.py`.

### `--resume-from` fails with "No 'syllabus/' subdirectory found"

`--resume-from` expects a previous **run directory** (e.g.
`output/2026-08-22_153000_ML_Basics`), **not** the `intake_session.json` file.

```bash
# Correct — pass the run directory
python src/main.py "ML Basics" --resume-from output/2026-08-22_153000_ML_Basics

# To reuse a saved intake session instead, use --load-session
python src/main.py "ML Basics" --load-session output/2026-08-22_153000_ML_Basics/intake_session.json
```

---

## License

MIT