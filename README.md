# syllabus-swarm 🐝

[![CI](https://github.com/loos-glr/syllabus-swarm/actions/workflows/ci.yml/badge.svg)](https://github.com/loos-glr/syllabus-swarm/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

> A local multi-agent workspace for generating vocational-level software development curriculum.

**syllabus-swarm** uses AI agents (powered by [CrewAI](https://www.crewai.com/)) connected to specialized language models via **OpenRouter** to collaboratively design, develop, and package complete course materials — syllabi, interactive theory artifacts, tiered coding labs, evaluation rubrics, and video-as-code compositions.

---

## Quick Start

```bash
# 1. Clone the repository
git clone git@github.com:loos-glr/syllabus-swarm.git
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

syllabus-swarm is built on **nine specialized AI agents**, each assigned a model optimized for its specific role:

| Agent | Role | Default Model | Rationale |
|---|---|---|---|
| **Intake Specialist** | Interviews the user to extract technical and pedagogical requirements mapped to Dutch SBB Kwalificatiedossiers | `deepseek/deepseek-v4-pro` | Strong reasoning for synthesising rich course context from user answers, with deep knowledge of MBO4 vocational education pathways (BOL/BBL) and kerntaken (P1-K1 through P4-K1). |
| **Curriculum Architect** | Designs syllabi using the Humanics framework (data literacy, technological literacy, human literacy) + experiential learning | `deepseek/deepseek-v4-pro` | State-of-the-art multi-step reasoning for crafting logically coherent, pedagogically sound syllabi that span weeks of content across three integrated literacies. |
| **Education Director** | Audits syllabi for time-budget math, realistic MBO4 workloads, and scheduling contradictions before content generation proceeds | `deepseek/deepseek-v4-pro` | Structured analysis and precise feasibility calculations to ensure every syllabus is deliverable within real classroom constraints. |
| **Theory Instructor** | Transforms abstract syllabus concepts into interactive learning artifacts (HTML/JS visualizations, pausing terminal scripts, Mermaid.js diagrams) | `deepseek/deepseek-v4-pro` | Strong writing + code generation for producing self-contained, runnable interactive artifacts that vocational students can engage with before starting hands-on labs. |
| **Lab & Project Developer** | Generates tiered hands-on coding exercises with starter code and fully-commented solution keys | `openrouter/qwen/qwen3-coder` | Purpose-built for programming tasks — produces cleaner starter code, more idiomatic solutions, and fewer hallucinated API calls than general-purpose models. |
| **QA Reviewer** | Reviews all generated labs for technical correctness (syntax, imports, runnability) and MBO4 didactic appropriateness | `openrouter/qwen/qwen3-coder` | Purpose-built for code understanding and review — catches syntax errors, missing imports, hallucinated variables, and didactic issues before they reach students. |
| **Media Strategist** | Analyzes curriculum module complexity and routes each module to the optimal instructional modality (text, interactive web, terminal CLI, or Video-as-Code) | `deepseek/deepseek-v4-pro` | Strong pedagogical reasoning for calibrating complexity scores and producing well-justified `ModalityDecision` outputs that drive the entire downstream generation pipeline. |
| **Video Engineer** | Generates deterministic temporal code (React/Remotion JSX) for educational video compositions — pure structural data, no prose | `deepseek/deepseek-v4-pro` | Balances creative temporal animation design with correct, runnable TypeScript/JSX output; produces `RemotionManifest` descriptors for version-controllable video compositions. |
| **Output Exporter** | Compiles and packages all materials into clean directory structures and a consolidated manifest | `deepseek/deepseek-v4-flash-latest` | Low-latency, low-cost completions ideal for manifest generation, file assembly, and Markdown packaging — reliability without burning reasoning-token budgets. |

### Pipeline & Output Structure

```
output/
└── <run_id>/                     # e.g. 2026-09-22_153000_Python_Basics
    ├── intake_session.json       # Saved intake interview (auto-saved)
    ├── _generation_state.json    # Resume/restart progress snapshot
    ├── README.md                 # Consolidated output manifest
    ├── course_graph.json         # Machine-readable course metadata
    ├── syllabus/                 # Curriculum Architect output
    ├── theory/                   # Theory Instructor artifacts (HTML/JS/CLI)
    ├── labs/                     # Lab Developer exercises & QA reports
    └── rubrics/                  # Evaluation rubrics
```

### Pydantic Models

| Model | Location | Purpose |
|---|---|---|
| `CourseSpecification` | `src/main.py` | Structured output from Intake Specialist |
| `IntakeSession` | `src/main.py` | Serializable record of a completed intake interview |
| `CourseGraph` | `src/models.py` | Machine-readable course metadata — composes `CourseSpecification` |
| `ModuleSummary` | `src/models.py` | Lightweight per-module record |
| `ModalityType` | `src/models.py` | Enum: `CLASSIC_READER`, `INTERACTIVE_WEB`, `INTERACTIVE_CLI`, `VIDEO_AS_CODE` |
| `ModalityDecision` | `src/models.py` | Media Strategist routing output with complexity score (0.0–1.0) |
| `RemotionManifest` | `src/models.py` | Deterministic VaC composition descriptor |
| `TierState` | `src/models.py` | Per-tier lab completion status for resume tracking |
| `GenerationState` | `src/models.py` | Full pipeline progress snapshot for `--resume-from` |
---

## Usage

```bash
# Full pipeline: intake + syllabus + theory + labs + QA review + manifest
python src/main.py "Data Science with Python"

# Syllabus only (skip theory, lab generation, and QA review)
python src/main.py "Full-Stack Web Development" --skip-labs

# Load a cohort profile (pre-populates static constraints, skips intake)
python src/main.py "Laravel Web Development" --profile config/profiles/program1_profile.yaml

# Resume a previous run (skip intake, re-run agents from last checkpoint)
python src/main.py "ML Basics" --resume-from output/2026-08-22_153000_ML_Basics

# Load a saved intake session (skip interactive interview)
python src/main.py "Advanced PHP" --load-session output/2026-08-22_153000_ML_Basics/intake_session.json

# Chain modules: inject prerequisites from a previous course
python src/main.py "Period 3 Project" --builds-upon output/2026-08-22_153000_ML_Basics

# Interactive prompt
python src/main.py
```
---

## Modality Routing & Video-as-Code (VaC)

syllabus-swarm features a **polyglot generation pipeline** that routes each curriculum module to the most effective instructional modality:

| Modality | Generator | Output Format | Best For |
|---|---|---|---|
| `CLASSIC_READER` | Theory Instructor | Markdown / text | Syntax, terminology, reference material |
| `INTERACTIVE_WEB` | Theory Instructor | Self-contained HTML/JS | Visual algorithms, state machines, data structures |
| `INTERACTIVE_CLI` | Theory Instructor | Pausing terminal scripts | CLI workflows, API walkthroughs, ETL pipelines |
| `VIDEO_AS_CODE` | Video Engineer | React/Remotion `.tsx` | Recursion, network protocols, temporal animations |

The **Media Strategist** evaluates each module's pedagogical complexity and outputs a `ModalityDecision` with a calibrated complexity score (0.0–1.0). The swarm state machine reads this decision and routes generation to either the Theory Instructor or the Video Engineer.

The **Video Engineer** produces deterministic `RemotionManifest` descriptors — pure structural data (scene sequences, timing, React component trees). Compositions are exported as valid `.tsx` files to `src/export/vac/`.

---

## Human-in-the-Loop (HITL) Feedback

```bash
# Run with interactive feedback prompt
python src/main.py "Python Basics"
# After generation completes, you'll be prompted to:
#   [A]pprove → proceed to export
#   [F]eedback → revise content and regenerate
#   [Q]uit → stop execution
```

The `SwarmState` state machine tracks execution through `GENERATING` → `AWAITING_FEEDBACK` → `EXPORTING` phases. Feedback is injected into the next Lab Developer iteration for incremental refinement.

---

## Resume & Generation State

The **GenerationState** system enables reliable pipeline resumption:

- `_generation_state.json` written after each pipeline stage completes
- Tracks per-tier lab completion, theory, syllabus review, and QA review
- `--resume-from` reads the state file and skips already-completed stages
- Agent iteration limits (`max_iter`) and rate limits (`max_rpm`) are configurable per-agent (see `.env.example`)
---

## Model Configuration

Every agent obtains its LLM through `src/llm_factory.py` resolving model, temperature, max_iter, and max_rpm through a **3-tier fallback chain**:

1. **Per-agent override** — `AGENT_{ROLE}_{PROPERTY}` (most specific)
2. **Agent-wide default** — `AGENT_DEFAULT_{PROPERTY}` (catch-all)
3. **Hardcoded sensible defaults** — baked into the factory

```bash
# See the resolved configuration for every agent
python -m src.llm_factory

# Override a specific agent's model
AGENT_CURRICULUM_ARCHITECT_MODEL=anthropic/claude-sonnet-4 python src/main.py "Python Basics"

# Bump QA Reviewer's iteration limit for large output sets
AGENT_QA_REVIEWER_MAX_ITER=250 python src/main.py "Data Science"
```

> See [OpenRouter Models](https://openrouter.ai/models) for available model identifiers.

---

## Configuration System

```
config/
├── school_defaults.yaml          # Immutable institution-wide defaults
└── profiles/
    ├── _TEMPLATE.yaml
    ├── program1_profile.yaml     # PHP/Laravel — BOL Pathway (Year 2)
    ├── scripting2_profile.yaml   # Python scripting — BBL Pathway
    ├── proces2_profile.yaml      # Process & Django
    ├── beroeps2_profile.yaml     # Professional Development
    ├── immersive_design_profile.yaml           # Immersive Design (WebXR & Unity)
    ├── immersive_design_blok1_profile.yaml     # Immersive Design Block 1
    └── vmbo_orientatie_mvi.yaml  # VMBO Orientation MVI
```

### Layering Model

```
1. school_defaults.yaml     ←  Institution-wide (MBO4 standards, O/V/G grading)
         │
2. cohort profile            ←  Per-track overrides (year level, tech stack)
         │
3. Intake Specialist         ←  Interactive interview fills remaining gaps
```

---

## Project Structure

```
syllabus-swarm/
├── src/
│   ├── agents/                    # Nine specialized CrewAI agents
│   ├── crews/syllabus_crew.py     # Pipeline orchestration & resume logic
│   ├── exporters/                 # File I/O, manifests, validation
│   ├── tasks/                     # CrewAI task definitions
│   ├── config_loader.py           # YAML profile loader
│   ├── llm_factory.py             # LLM factory with 3-tier fallback
│   ├── main.py                    # CLI entry point & HITL loop
│   └── models.py                  # Pydantic domain models
├── config/profiles/               # Cohort-specific profiles
├── docs/                          # Architecture & constitution
├── tests/                         # pytest suite (504 tests)
├── pyproject.toml                 # Ruff settings
└── .env.example                   # Environment variable template
```

---

## Troubleshooting

### Python version errors

The project requires **Python 3.12+**. If you see import errors about `datetime.UTC` or `crewai`:

```bash
source .venv/bin/activate
python3.12 -m venv .venv  # recreate if needed
```

### "Maximum iterations reached"

Increase the relevant agent's iteration limit:

```bash
export AGENT_QA_REVIEWER_MAX_ITER=250
export AGENT_DEFAULT_MAX_ITER=100
```

### `--resume-from` fails

Pass the **run directory** (not `intake_session.json`):

```bash
# Correct
python src/main.py "ML Basics" --resume-from output/2026-08-22_153000_ML_Basics

# For reusing a saved intake session only
python src/main.py "ML Basics" --load-session output/2026-08-22_153000_ML_Basics/intake_session.json
```

---

## Development

```bash
# Run tests (504 tests)
pytest

# Lint and format
ruff check .
ruff format .
```

### Code Style

- **Ruff** for formatting and linting (`line-length = 100`, `quote-style = "double"`)
- `from __future__ import annotations` in every module
- Modern type hints: `list[str]`, `dict[str, int]`, `str | None`

---

## License

MIT