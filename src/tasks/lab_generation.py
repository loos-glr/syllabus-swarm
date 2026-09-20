"""
lab_generation.py — Tiered Lab Generation Task (Humanics-Aligned)
=================================================================

Issue #3: Core Agent — The Lab & Project Developer (Tiered Coding Challenges)

Defines a CrewAI **Task** that, when executed by the Lab Developer
agent, produces a complete set of tiered coding labs derived from a
course syllabus.  The task specification:

  • Generates labs across **three progressive tiers** — Foundations,
    Application, and Architecture — each building on the previous.
  • Produces **starter/** scaffolding with TODO markers and **solution/**
    reference implementations for every lab.
  • Includes a **README.md** per lab with learning objectives and key
    concepts.
  • Writes output to ``output/labs/<course_name>/<tier_name>/starter/``
    and ``output/labs/<course_name>/<tier_name>/solution/``.
  • Ensures every lab is **self-contained** — a student can clone and
    run it in isolation with a single command.
  • Dynamically adapts file extensions, linters, and tooling references
    based on the ``language`` parameter (e.g. ``.py`` / ``ruff`` for
    Python, ``.js`` / ``eslint`` for JavaScript).
"""

from __future__ import annotations

from crewai import Agent, Task

# ---------------------------------------------------------------------------
# Language configuration — maps a language name to its tooling defaults.
# ---------------------------------------------------------------------------

_LANGUAGE_CONFIG: dict[str, dict[str, str]] = {
    "Python": {
        "ext": ".py",
        "linter": "ruff check",
        "type_checker": "mypy --strict",
        "dep_file": "requirements.txt",
        "build_file": "pyproject.toml",
        "lang_version": "Python ≥3.10",
        "run_cmd": "python",
    },
    "JavaScript": {
        "ext": ".js",
        "linter": "eslint",
        "type_checker": "tsc --noEmit  # if using JSDoc types",
        "dep_file": "package.json",
        "build_file": "package.json",
        "lang_version": "Node.js ≥18 LTS",
        "run_cmd": "node",
    },
    "TypeScript": {
        "ext": ".ts",
        "linter": "eslint",
        "type_checker": "tsc --noEmit",
        "dep_file": "package.json",
        "build_file": "tsconfig.json",
        "lang_version": "Node.js ≥18 LTS + TypeScript ≥5.0",
        "run_cmd": "npx ts-node",
    },
    "Java": {
        "ext": ".java",
        "linter": "checkstyle",
        "type_checker": "javac -Xlint:all",
        "dep_file": "pom.xml  # or build.gradle",
        "build_file": "pom.xml",
        "lang_version": "Java ≥17 LTS",
        "run_cmd": "java",
    },
    "Go": {
        "ext": ".go",
        "linter": "golangci-lint run",
        "type_checker": "go vet",
        "dep_file": "go.mod",
        "build_file": "go.mod",
        "lang_version": "Go ≥1.21",
        "run_cmd": "go run",
    },
    "Rust": {
        "ext": ".rs",
        "linter": "clippy",
        "type_checker": "cargo check",
        "dep_file": "Cargo.toml",
        "build_file": "Cargo.toml",
        "lang_version": "Rust ≥1.75 (stable)",
        "run_cmd": "cargo run",
    },
    "C#": {
        "ext": ".cs",
        "linter": "dotnet format",
        "type_checker": "dotnet build /warnaserror",
        "dep_file": "*.csproj",
        "build_file": "*.csproj",
        "lang_version": ".NET ≥8.0",
        "run_cmd": "dotnet run",
    },
    "PHP": {
        "ext": ".php",
        "linter": "phpcs",
        "type_checker": "phpstan analyse",
        "dep_file": "composer.json",
        "build_file": "composer.json",
        "lang_version": "PHP ≥8.2",
        "run_cmd": "php",
    },
}


def _get_lang_config(language: str) -> dict[str, str]:
    """Return the tooling config for *language*, falling back to Python."""
    # Normalise: strip whitespace, title-case for matching.
    key = language.strip()
    if key in _LANGUAGE_CONFIG:
        return _LANGUAGE_CONFIG[key]
    # Try case-insensitive match.
    for name, cfg in _LANGUAGE_CONFIG.items():
        if name.lower() == key.lower():
            return cfg
    # Fallback to Python with a warning-like default.
    return _LANGUAGE_CONFIG["Python"]


# ---------------------------------------------------------------------------
# Directory-structure requirement — injected into the description so the
# agent produces output organised in the mandated layout.
# ---------------------------------------------------------------------------
_DIRECTORY_STRUCTURE_REQUIREMENT: str = (
    "## 📁  Directory Structure Requirements\n\n"
    "Your output must create the following directory layout under "
    "`output/labs/<course_name>/`.  The placeholders `<course_name>` and "
    "`<tier_name>` must be replaced with the actual values:\n\n"
    "```\n"
    "output/labs/<course_name>/\n"
    "│\n"
    "├── tier1_foundations/\n"
    "│   ├── starter/\n"
    "│   │   ├── README.md          # Learning objectives, key concepts,\n"
    "│   │   │                       # setup instructions, and pre-requisites\n"
    "│   │   └── <lab_files>{ext}     # Scaffolded code with TODO markers\n"
    "│   └── solution/\n"
    "│       ├── README.md          # Same as starter, plus solution notes\n"
    "│       └── <lab_files>{ext}     # Fully-working, commented code\n"
    "│\n"
    "├── tier2_application/\n"
    "│   ├── starter/\n"
    "│   │   ├── README.md\n"
    "│   │   └── <multi-file project scaffold>\n"
    "│   └── solution/\n"
    "│       ├── README.md\n"
    "│       └── <multi-file project with full implementation>\n"
    "│\n"
    "└── tier3_architecture/\n"
    "    ├── starter/\n"
    "    │   ├── README.md\n"
    "    │   └── <system-design / microservices scaffold>\n"
    "    └── solution/\n"
    "        ├── README.md\n"
    "        └── <system-design / microservices full implementation>\n"
    "```\n\n"
    "**Key rules:**\n"
    "- Every lab directory (starter and solution) MUST contain a README.md.\n"
    "- Starter files MUST contain `# TODO:` markers — at least 3 per file —\n"
    '  that a student can search for with `grep -r "TODO"`.  Each TODO must\n'
    "  be a concrete, actionable instruction (never a placeholder).\n"
    "- Solution files MUST be complete, runnable, and thoroughly commented\n"
    "  so a self-study learner can understand every line.\n"
    "- Labs MUST be self-contained: a student should be able to `cd` into\n"
    "  any starter/ or solution/ directory, follow the README, and run the\n"
    "  code with zero external dependencies beyond what is documented.\n"
    "- File names and directory names MUST use snake_case.\n"
    "- All source files MUST use the `{ext}` extension for {language}.\n"
)
# ---------------------------------------------------------------------------
# Tier definitions — injected into the description as non-negotiable scope.
# ---------------------------------------------------------------------------
_TIER_DEFINITIONS: str = (
    "## 🔺  Tier Definitions (Non-Negotiable)\n\n"
    "### Tier 1 — Foundations\n"
    "**Focus:** Syntax drills, basic patterns, and single-file exercises.\n\n"
    "- Each lab must be a single `{ext}` file (plus README).\n"
    "- At least 3 labs for this tier.\n"
    "- Topics: variables, control flow, functions, data structures (lists,\n"
    "  dicts/objects, sets), file I/O, error handling, and basic OOP.\n"
    "- Code complexity: ≤ 150 lines per solution file.\n"
    "- Every lab must include at least one exercise that reads input and\n"
    "  produces output (stdin/stdout or file-based).\n"
    "- Humanics tags: `[T]` on every file, `[D]` on data-handling exercises,\n"
    "  `[H]` in README reflections on code readability and collaboration.\n\n"
    "### Tier 2 — Application\n"
    "**Focus:** Multi-file projects, APIs, and data processing.\n\n"
    "- Each lab must span 3–5 files (plus README) organised as a coherent\n"
    "  mini-project.\n"
    "- At least 3 labs for this tier.\n"
    "- Topics: REST API consumption, database CRUD, data pipelines (ETL),\n"
    "  CLI tools, web scraping, testing, and packaging.\n"
    "- Must include a `{dep_file}` listing all dependencies.\n"
    "- Code complexity: ≤ 500 lines total across all solution files.\n"
    "- Humanics tags: `[T]` on tooling/integration, `[D]` on every\n"
    "  data-processing step, `[H]` in README sections on API ethics,\n"
    "  data privacy, and team-worthy code practices.\n\n"
    "### Tier 3 — Architecture\n"
    "**Focus:** System design, microservices, and deployment.\n\n"
    "- Each lab must span 5–10 files representing a service-oriented\n"
    "  architecture.\n"
    "- At least 3 labs for this tier.\n"
    "- Topics: message queues (RabbitMQ/Redis), containerisation (Docker),\n"
    "  service orchestration (docker-compose), database sharding/replication,\n"
    "  CI/CD pipelines, cloud deployment (AWS/GCP/Azure basics), and\n"
    "  observability (logging, metrics, health checks).\n"
    "- Must include a `Dockerfile` and `docker-compose.yml` so the entire\n"
    "  system can be started with `docker-compose up`.\n"
    "- Must include a `Makefile` or shell script for common operations.\n"
    "- Code complexity: ≤ 1500 lines total across all solution files.\n"
    "- Humanics tags: `[T]` on every service, `[D]` on monitoring/metrics\n"
    "  dashboards, `[H]` in README sections on system ethics, accessibility,\n"
    "  and incident-response communication.\n"
)
# ---------------------------------------------------------------------------
# README template mandate — injected into the description.
# ---------------------------------------------------------------------------
_README_TEMPLATE_REQUIREMENT: str = (
    "## 📄  README.md Requirements (Per Lab)\n\n"
    "Every starter/ and solution/ directory MUST contain a README.md with "
    "the following exact sections:\n\n"
    "```markdown\n"
    "# {{Lab Title}}\n\n"
    "## 🎯 Learning Objectives\n"
    "- Bullet list of 3–5 specific, measurable learning outcomes.\n\n"
    "## 🧠 Key Concepts\n"
    "- Bullet list of concepts this lab teaches or reinforces.\n\n"
    "## 📋 Prerequisites\n"
    "- What the student must already know or have installed.\n"
    "- Include exact language version ({lang_version}) and any packages.\n\n"
    "## 🚀 Getting Started\n"
    "```bash\n"
    "# Exact commands to clone, install deps, and run.\n"
    "# Must work when copy-pasted into a fresh terminal.\n"
    "```\n\n"
    "## 🏗️ Project Structure\n"
    "```\n"
    "# Tree view of all files with brief descriptions.\n"
    "```\n\n"
    "## 📝 Lab Instructions\n"
    "# Step-by-step walkthrough of what to build.\n"
    "# In starter/ this is the main guide.\n"
    '# In solution/ add a "Solution Walkthrough" sub-section.\n\n'
    "## 🔍 Humanics Reflection  [H]\n"
    "# Prompts that ask the student to reflect on:\n"
    "# - How does this code impact people?\n"
    "# - What ethical considerations apply?\n"
    "# - How would you explain this to a non-technical colleague?\n\n"
    "## 📚 Additional Resources\n"
    "# Links to docs, tutorials, and further reading.\n"
    "```\n"
)

# ---------------------------------------------------------------------------
# Humanics embedding mandate — injected verbatim into the description.
# ---------------------------------------------------------------------------
_HUMANICS_LAB_MANDATE: str = (
    "## 🔴 MANDATORY — Humanics Literacies in Labs\n\n"
    "Every lab you produce must visibly integrate **three inseparable "
    "literacies** grounded in Joseph Aoun's Humanics framework:\n\n"
    "- **Technological Literacy [T]** — Every starter file must require "
    "the student to write, debug, and run code.  Every solution file must "
    "demonstrate idiomatic, well-structured code with meaningful comments.  "
    "Include tooling fluency (git commands, virtual environments, linters).\n"
    "- **Data Literacy [D]** — At least one lab per tier must involve "
    "collecting, processing, analysing, or visualising data.  Include "
    "exercises that require the student to make a data-informed decision "
    "and justify it in a comment or README section.\n"
    "- **Human Literacy [H]** — Every README.md must include a **Humanics "
    "Reflection** section (see README template above) that prompts the "
    "student to consider ethics, accessibility, team dynamics, or societal "
    "impact.  Tier 3 labs must also address incident response and "
    "responsible deployment.\n\n"
    "These are **not** separate modules — they are threads woven into "
    "every lab.  Tag relevant sections with `[T]`, `[D]`, or `[H]` so "
    "instructors can verify coverage at a glance.\n"
)

# ---------------------------------------------------------------------------
# Code-quality mandate — injected into the description.
# ---------------------------------------------------------------------------
_CODE_QUALITY_MANDATE: str = (
    "## 🔴 MANDATORY — Code Quality Standards\n\n"
    "- All {language} code must use **type hints** where the language "
    "supports them ({lang_version} syntax).\n"
    "- Every `{ext}` file must start with a module-level docstring/comment "
    "explaining its purpose.\n"
    "- Functions must have docstrings/JSDoc (at least a one-liner for simple "
    "functions, full documentation for complex ones).\n"
    "- Solution code must pass `{linter}` and `{type_checker}` with zero "
    "errors (unless the lab *teaches* fixing those errors).\n"
    "- TODO markers must follow the format: `# TODO(<tier>): <actionable "
    "instruction>` — e.g. `# TODO(t1): Implement the binary_search "
    "function to return -1 when the target is not found.`\n"
    '- Never leave a TODO that says "implement this" without specifying '
    "exactly *what* to implement and *how* to verify correctness.\n"
)

# ---------------------------------------------------------------------------
# Tool-usage mandate — forces the agent to use output_export_tool iteratively.
# ---------------------------------------------------------------------------
_TOOL_USAGE_MANDATE: str = (
    "## 🔴 MANDATORY — Use the `output_export_tool` to Write Files\n\n"
    "You have access to the **`output_export_tool`** (specifically the "
    "`write-labs` command).  You MUST use this tool to generate and save "
    "the actual code files for Tier 1, Tier 2, and Tier 3 to disk.\n\n"
    "**Workflow:**\n"
    "1. Read the ``run_id`` from the course context provided at the start "
    "of this task description (look for ``**Run ID:**``).  You MUST include "
    "this ``run_id`` in EVERY tool call — this ensures files land in the "
    "correct per-run directory and do NOT leak into the shared global "
    "output folder.\n"
    "2. Generate the labs for **Tier 1** first.  Call `output_export_tool` "
    "with `command='write-labs'` plus the keyword arguments `course_name`, "
    "`tier` (e.g. `'tier1_foundations'`), `files` (a dict mapping relative "
    "paths to file contents), and `run_id`.\n"
    "3. After Tier 1 files are written, move to **Tier 2** and repeat.\n"
    "4. After Tier 2, move to **Tier 3** and repeat.\n"
    "5. Once ALL files for ALL tiers are successfully written to disk, "
    "produce your final textual response: a complete, well-structured "
    "Markdown README that serves as the top-level index for all labs.  "
    "It must include the course title, a brief overview, and a numbered "
    "list of every lab with its tier and a one-line description.\n\n"
    "**Calling convention (keyword arguments, NOT a JSON string):**\n"
    "Pass each value as a separate keyword argument to the tool.  For "
    "`files`, pass a mapping (dictionary) of relative paths to file "
    "contents.  Example tool call:\n"
    "```\n"
    "command = 'write-labs'\n"
    "course_name = 'Javascript OOP Basics'\n"
    "tier = 'tier1_foundations'\n"
    "run_id = '2026-08-23_120000_Javascript_OOP_Basics'\n"
    "files = {{\n"
    "    'starter/README.md': '# Lab 1: Variables and Data Types\\n\\n"
    "## 🎯 Learning Objectives\\n...',\n"
    "    'starter/lab1_variables{ext}': '// TODO(t1): Declare a constant...',\n"
    "    'solution/README.md': '# Lab 1 Solution: Variables and Data Types"
    "\\n\\n...',\n"
    "    'solution/lab1_variables{ext}': '// Complete working solution with "
    "comments\\nconst PI = 3.14159;\\n...',\n"
    "}}\n"
    "```\n\n"
    "**Do NOT** attempt to write all file contents inline in your final "
    "text response.  Use the tool for every file.  Your final text response "
    "must be a real, detailed top-level README index — NOT a placeholder "
    'like "No more tool calls" or "Just the final answer".\n'
)

# ---------------------------------------------------------------------------
# Task factory
# ---------------------------------------------------------------------------


def create_lab_generation_task(
    *,
    course_name: str,
    agent: Agent,
    syllabus_context: str | None = None,
    topic_focus: str | None = None,
    language: str = "Python",
    run_id: str | None = None,
    tier: str | None = None,
    human_feedback: str | None = None,
    verbose: bool = False,
) -> Task:
    """Create a CrewAI Task that generates tiered coding labs from a syllabus.

    Parameters
    ----------
    course_name : str
        The name of the vocational course (e.g. "Full-Stack Web Development").
    agent : Agent
        The Lab Developer agent that will execute this task.
    syllabus_context : str or None
        Optional syllabus content to ground lab generation.  When provided the
        agent extracts topics, modules, and learning objectives from this
        context to design relevant labs.  When omitted the agent infers lab
        topics from the course name.
    topic_focus : str or None
        Optional comma-separated list of specific topics to emphasise
        (e.g. "recursion, graph algorithms, REST APIs").
    language : str
        The primary programming language for the labs.  Defaults to
        ``"Python"``.  Determines file extensions, linters, type checkers,
        and dependency-file names used throughout the task instructions.
    run_id : str or None
        The unique run identifier (e.g. ``"2026-08-23_120000_Course_Name"``).
        When provided, it is injected into the task description so the agent
        can pass it through to every ``write-labs`` tool call, ensuring files
        land in the correct per-run output directory.
    tier : str or None
        When provided (e.g. ``"tier1_foundations"``), limits the task to
        generating labs for that single tier only.  This reduces the prompt
        size and allows the LLM to focus on one tier at a time.  When
        ``None``, generates all three tiers (legacy behaviour).
    verbose : bool
        Enable detailed task execution logging.

    Returns
    -------
    Task
        Fully-configured CrewAI Task ready to be assigned to a Crew.
    """
    # ── Resolve language-specific tooling ──────────────────────────────
    cfg = _get_lang_config(language)
    ext = cfg["ext"]
    linter = cfg["linter"]
    type_checker = cfg["type_checker"]
    dep_file = cfg["dep_file"]
    build_file = cfg["build_file"]
    lang_version = cfg["lang_version"]
    run_cmd = cfg["run_cmd"]

    # ── Format the static mandates with language-specific values ────────
    fmt = dict(
        language=language,
        ext=ext,
        linter=linter,
        type_checker=type_checker,
        dep_file=dep_file,
        build_file=build_file,
        lang_version=lang_version,
        run_cmd=run_cmd,
    )

    directory_structure = _DIRECTORY_STRUCTURE_REQUIREMENT.format(**fmt)
    tier_definitions = _TIER_DEFINITIONS.format(**fmt)
    readme_template = _README_TEMPLATE_REQUIREMENT.format(**fmt)
    code_quality = _CODE_QUALITY_MANDATE.format(**fmt)
    tool_usage = _TOOL_USAGE_MANDATE.format(**fmt)

    # ── Determine tier scope ───────────────────────────────────────────
    if tier:
        # Single-tier mode: extract only the relevant tier definition.
        tier_label = {
            "tier1_foundations": "Tier 1 — Foundations",
            "tier2_application": "Tier 2 — Application",
            "tier3_architecture": "Tier 3 — Architecture",
        }.get(tier, tier)

        # Extract just the relevant tier section from the full definitions.
        tier_section = ""
        for line in tier_definitions.split("\n"):
            if line.startswith("### ") and tier_label not in line and tier_section:
                break
            if tier_section or line.startswith(f"### {tier_label}"):
                tier_section += line + "\n"

        tier_scope = (
            f"## 🔺  Scope — {tier_label} ONLY\n\n"
            f"You are generating labs for **{tier_label}** only.  "
            f"Do NOT generate labs for other tiers.\n\n"
            f"{tier_section}\n"
            f"**Course-adaptation guidance:** If the syllabus topics do "
            f"not naturally extend to this tier's typical depth (e.g. a "
            f"foundational OOP course may not cover Docker or "
            f"microservices), adapt the labs to bridge the gap.  For "
            f"Tier 3 in a basic course, a single-file GitHub Actions "
            f"workflow or a simple `docker-compose.yml` wrapping the "
            f"Tier 2 project is perfectly acceptable.  The goal is to "
            f"expose students to the concepts, not to go deep into "
            f"infrastructure.  Produce at least 2 lab projects for this "
            f"tier even if simplified.\n"
        )
    else:
        tier_scope = (
            "\n\nYour task is to design a complete set of self-contained coding "
            "labs organised into three progressive tiers.  Each tier must have "
            "**at least 3 labs**, and every lab must include both a **starter/** "
            "scaffold and a **solution/** reference implementation.\n\n"
            f"{tier_definitions}\n"
        )

    # ---- Build the description ------------------------------------------
    description_parts: list[str] = [
        f"Generate tiered coding labs for the following course:\n\n"
        f"**Course Name:** {course_name}\n"
        f"**Primary Language:** {language}\n",
    ]

    if run_id:
        description_parts.append(
            f"\n**Run ID:** {run_id}\n"
            f"(MUST include this ``run_id`` in every `write-labs` tool call)\n"
        )

    if syllabus_context:
        # Truncate syllabus to ~3000 chars to keep prompt manageable.
        ctx = syllabus_context
        if len(ctx) > 4000:
            ctx = ctx[:4000] + "\n\n[... syllabus truncated for length ...]\n"
        description_parts.append(f"\n**Syllabus Context:**\n\n{ctx}\n")

    if topic_focus:
        description_parts.append(f"\n**Topic Focus:** {topic_focus}\n")

    description_parts.append(
        f"\n\n{tier_scope}\n"
        f"{tool_usage}\n\n"
        f"{directory_structure}\n\n"
        f"{readme_template}\n\n"
        f"{_HUMANICS_LAB_MANDATE}\n\n"
        f"{code_quality}\n"
    )

    description = "".join(description_parts)

    # ── Inject human feedback (HITL loop) ───────────────────────────
    if human_feedback:
        description += (
            f"\n\n## ⚠️ Human Feedback (Instructor Review)\n\n"
            f"The following feedback was provided by a human reviewer "
            f"and MUST be addressed in this iteration:\n\n"
            f"{human_feedback}\n\n"
            f"Please revise the output to incorporate this feedback "
            f"while maintaining all other requirements.\n"
        )

    # ---- Build the expected_output --------------------------------------
    if tier:
        expected_output = (
            f"## 🔴 CRITICAL: You MUST use the `output_export_tool`\n\n"
            f'Use the `output_export_tool` with `command="write-labs"` to '
            f"write ALL lab files for **{tier}** to disk.  Include at least "
            f"3 labs with starter/ and solution/ directories.\n\n"
            f"**Once all files are written**, produce a Markdown summary "
            f"listing each lab with its tier and a one-line description.\n"
        )
    else:
        expected_output = (
            "## 🔴 CRITICAL: You MUST use the `output_export_tool`\n\n"
            "You have access to the `output_export_tool` with the `write-labs` "
            "command.  You MUST use this tool to generate and save the actual "
            "code files for Tier 1, Tier 2, and Tier 3 to disk.\n\n"
            "**Once all files are successfully written**, your FINAL textual "
            "response must be a complete, well-structured Markdown README that "
            "serves as the top-level index for all labs.\n"
        )

    # ---- Compute the output file path -----------------------------------
    if run_id:
        output_file: str = f"output/{run_id}/labs/README.md"
    else:
        output_file: str = "output/labs/README.md"

    # ---- Assemble and return the Task -----------------------------------
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
        output_file=output_file,
        async_execution=False,
    )


# ---------------------------------------------------------------------------
# Convenience alias — mirrors the naming convention expected by crews.
# ---------------------------------------------------------------------------
create_lab_task = create_lab_generation_task

# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.agents.curriculum_architect import create_curriculum_architect

    # Use the existing architect agent for self-test; in production a
    # dedicated Lab Developer agent would be used.
    agent = create_curriculum_architect(verbose=True)
    task = create_lab_generation_task(
        course_name="Python for Data Engineering",
        agent=agent,
        topic_focus="ETL pipelines, REST APIs, Docker, PostgreSQL",
        language="Python",
        verbose=True,
    )
    print("✅ Lab Generation Task created successfully.\n")
    print(f"   Output file:  {task.output_file}")
    print(f"   Agent role:   {agent.role.split(chr(10))[0]}")
    print(f"   Async:        {task.async_execution}")
    print(f"   Description length: {len(task.description)} chars")
    print(f"   Expected output length: {len(task.expected_output)} chars")
