"""
theory_generation.py — Theory Artifact Generation Task
======================================================

Defines a CrewAI **Task** that, when executed by the Theory Instructor
agent, produces interactive theory artifacts derived from a course
syllabus.  The task specification:

  • Reads the generated syllabus and identifies the core concept for
    each of the 3 tiers.
  • Generates exactly ONE theory artifact per tier using the format
    most appropriate for the concept (HTML interactive, terminal script,
    or Mermaid.js Markdown diagram).
  • Writes artifacts into a ``theory/`` subfolder inside each tier's
    lab directory (e.g. ``output/<run_id>/labs/<course>/tier1_foundations/theory/``),
    keeping theory and starter code bundled together for the student.
  • Uses the ``output_export_tool`` with ``command="write-directory-tree"``
    to persist files to disk.
"""

from __future__ import annotations

from crewai import Agent, Task

# ---------------------------------------------------------------------------
# Format selection guidance — helps the agent pick the right format per tier.
# ---------------------------------------------------------------------------
_FORMAT_SELECTION_GUIDE: str = (
    "## 🎨  Format Selection Guide\n\n"
    "For each tier, choose the ONE format that best fits the core concept:\n\n"
    "| Tier | Typical Concepts | Recommended Format |\n"
    "|------|-----------------|--------------------|\n"
    "| Tier 1 — Foundations | Syntax, control flow, data structures, basic OOP, "
    "algorithms | **Format A (HTML/JS)** — visualise sorting, searching, "
    "recursion, or state transitions |\n"
    "| Tier 2 — Application | APIs, data pipelines, CLI tools, testing, "
    "databases | **Format B (Terminal Script)** — step-by-step walkthrough "
    "of API calls, ETL flows, or CLI workflows |\n"
    "| Tier 3 — Architecture | Microservices, Docker, CI/CD, system design, "
    "observability | **Format C (Mermaid.js Markdown)** — sequence diagrams, "
    "class diagrams, deployment topologies |\n\n"
    "**Override rule:** If a tier's concept clearly fits a different format "
    "better, use your judgment.  The table above is a guideline, not a "
    "straitjacket.  For example, if Tier 1 covers REST APIs, a terminal "
    "script may be more appropriate than HTML.\n"
)

# ---------------------------------------------------------------------------
# Artifact requirements — what each format must include.
# ---------------------------------------------------------------------------
_ARTIFACT_REQUIREMENTS: str = (
    "## 📐  Artifact Requirements Per Format\n\n"
    "### Format A — Interactive HTML/JS (``.html``)\n"
    "- Single, self-contained file that opens in any browser.\n"
    "- Inline ``<style>`` for all CSS (no external stylesheets).\n"
    "- Inline ``<script>`` for all JavaScript (no external libraries unless "
    'absolutely essential; if needed, use a CDN ``<script src="...">``).\n'
    "- Clear title (``<h1>``) and learning objectives (``<ul>``) at the top.\n"
    "- Interactive controls: buttons, sliders, input fields, or draggable "
    "elements that let the student manipulate the visualisation.\n"
    "- Step-by-step mode where applicable (e.g. 'Next Step' button for "
    "algorithms).\n"
    "- Responsive layout that works on a 1366×768 screen (common laptop).\n"
    "- Comments in the JavaScript explaining key logic.\n\n"
    "### 🔴  Format A — CRITICAL Engineering Requirements\n\n"
    "These are NON-NEGOTIABLE.  A file that violates any of these rules "
    "will NOT work when a student opens it:\n\n"
    "1. **Place ``<script>`` at the END of ``<body>``** — NOT in ``<head>``.  "
    "This ensures all DOM elements exist before your code runs.  "
    "Alternatively, wrap ALL JavaScript in a ``DOMContentLoaded`` listener:\n"
    "   ```javascript\n"
    "   document.addEventListener('DOMContentLoaded', function() {\n"
    "       // ALL your code goes here\n"
    "   });\n"
    "   ```\n\n"
    "2. **Null-check EVERY DOM query** — Every ``document.getElementById()``, "
    "``document.querySelector()``, or ``document.querySelectorAll()`` call "
    "MUST be followed by a null/empty check before use.  If an element is "
    "missing, log a clear error and return early:\n"
    "   ```javascript\n"
    "   const el = document.getElementById('my-element');\n"
    "   if (!el) {\n"
    "       console.error('FATAL: #my-element not found in DOM');\n"
    "       return;\n"
    "   }\n"
    "   ```\n\n"
    "3. **Wrap initialization in try/catch** — The main ``render()`` or "
    "``init()`` function MUST be wrapped in a try/catch block that displays "
    "errors VISIBLY in the page (not just console.error):\n"
    "   ```javascript\n"
    "   try {\n"
    "       render();\n"
    "   } catch (err) {\n"
    "       document.body.innerHTML = '<div style=\"color:red;padding:2rem;\">'\n"
    "           + '<h2>⚠️ Initialization Error</h2>'\n"
    "           + '<pre>' + err.message + '</pre></div>';\n"
    "       console.error(err);\n"
    "   }\n"
    "   ```\n\n"
    "4. **Validate data references before use** — If your code references "
    "objects by ID (e.g. node IDs in a graph, step indices in an array), "
    "validate that the referenced item EXISTS before using it.  For example, "
    "when drawing edges between nodes, verify both source and target nodes "
    "exist in your data structure:\n"
    "   ```javascript\n"
    "   const fromNode = nodes.find(n => n.id === edge.from);\n"
    "   const toNode = nodes.find(n => n.id === edge.to);\n"
    "   if (!fromNode || !toNode) {\n"
    "       console.warn('Skipping edge: missing node', edge);\n"
    "       continue;\n"
    "   }\n"
    "   ```\n\n"
    "5. **Add a visible error banner area** — Include a hidden ``<div>`` at "
    "the top of the page that becomes visible when errors occur:\n"
    "   ```html\n"
    '   <div id="error-banner" style="display:none;background:#e94560;'
    'color:#fff;padding:1rem;font-family:monospace;"></div>\n'
    "   ```\n"
    "   Write a helper function ``showError(msg)`` that populates and "
    "displays this banner.\n\n"
    "6. **Use ``console.log`` at key initialization points** — Log when the "
    "script starts, when each major component initializes, and when the "
    "render completes.  This makes debugging trivial:\n"
    "   ```javascript\n"
    "   console.log('[init] Script starting...');\n"
    "   console.log('[init] DOM ready, found', document.querySelectorAll('button').length, 'buttons');\n"
    "   console.log('[render] Drawing', nodes.length, 'nodes and', edges.length, 'edges');\n"
    "   console.log('[render] Complete.');\n"
    "   ```\n\n"
    "7. **Test your own logic** — Before writing the final file, mentally "
    "trace through the data flow.  If you have a ``scenarios`` object with "
    "``steps`` arrays, verify that every ``step`` has all required fields "
    "before the render function tries to use them.  If you reference "
    "``stepData.nodes``, make sure ``stepData`` is not undefined first.\n\n"
    "### Format B — Pausing Terminal Script (``.sh``)\n"
    "- Single, self-contained shell script (bash).\n"
    "- Starts with ``#!/usr/bin/env bash`` and ``set -euo pipefail``.\n"
    "- Every section prints a clear heading (e.g. ``echo '=== Step 1: ... ==='``).\n"
    "- Pauses between steps with ``read -r -p 'Press Enter to continue...'``.\n"
    "- Explains *what* each command does and *why* before running it.\n"
    "- Handles errors gracefully (explain what went wrong if a command fails).\n"
    "- Includes a summary at the end recapping what was learned.\n"
    "- Must be runnable with ``bash theory.sh`` from the theory/ directory.\n\n"
    "### Format C — Mermaid.js Markdown (``.md``)\n"
    "- Valid Markdown with embedded Mermaid.js code blocks.\n"
    "- Starts with an H1 title and learning objectives.\n"
    "- Each diagram is in a fenced code block: `` ```mermaid ... ``` ``.\n"
    "- Explanatory prose between diagrams — not just a wall of diagrams.\n"
    "- At least 2 distinct diagrams per artifact (e.g. class diagram + "
    "sequence diagram).\n"
    "- Diagrams must use proper Mermaid syntax that renders correctly on "
    "GitHub, VS Code, and Mermaid Live.\n"
    "- Include a 'Key Takeaways' section at the bottom.\n"
)

# ---------------------------------------------------------------------------
# Tool-usage mandate — forces the agent to use output_export_tool.
# ---------------------------------------------------------------------------
_TOOL_USAGE_MANDATE: str = (
    "## 🔴 MANDATORY — Use the `output_export_tool` to Write Files\n\n"
    "You have access to the **`output_export_tool`**.  You MUST use the "
    "`write-directory-tree` command to write your theory artifacts to disk.\n\n"
    "**Workflow:**\n"
    "1. Read the ``run_id`` and ``course_name`` from the context provided "
    "at the start of this task description (look for ``**Run ID:**`` and "
    "``**Course Name:**``).  You MUST include these in EVERY tool call.\n"
    "2. Generate the theory artifact for **Tier 1** first.  Call "
    '`output_export_tool` with `command="write-directory-tree"`, passing '
    "`base_path` and `files` (a dict mapping relative paths to file contents).\n"
    "3. After Tier 1 files are written, move to **Tier 2** and repeat.\n"
    "4. After Tier 2, move to **Tier 3** and repeat.\n"
    "5. Once ALL files for ALL tiers are successfully written to disk, "
    "produce your final textual response: a Markdown summary listing each "
    "tier, the format chosen, the filename, and a one-line description of "
    "the artifact.\n\n"
    "**Base path for each tier:**\n"
    "```\n"
    "output/<run_id>/labs/<tier>/theory/\n"
    "```\n\n"
    "**Example tool call for Tier 1 (you MUST follow this exact structure):**\n"
    "```json\n"
    "{{\n"
    '  "command": "write-directory-tree",\n'
    '  "base_path": "output/2026-08-23_120000_Javascript_OOP/labs/tier1_foundations/theory",\n'
    '  "files": {{\n'
    '    "sorting_visualizer.html": "<!DOCTYPE html>\\n<html lang=\\"en\\">\\n..."\n'
    "  }}\n"
    "}}\n"
    "```\n\n"
    "**Do NOT** attempt to write all file contents inline in your final "
    "text response.  Use the tool for every file.  Your final text response "
    "must be a real summary of what was generated — NOT a placeholder.\n"
)

# ---------------------------------------------------------------------------
# Task factory
# ---------------------------------------------------------------------------


def create_theory_task(
    *,
    agent: Agent,
    course_name: str,
    syllabus_context: str | None = None,
    run_id: str | None = None,
    tier: str | None = None,
    material_language: str = "Dutch",
    verbose: bool = False,
) -> Task:
    """Create a CrewAI Task that generates interactive theory artifacts.

    Parameters
    ----------
    agent : Agent
        The Theory Instructor agent that will execute this task.
    course_name : str
        The name of the vocational course (e.g. "Javascript OOP").
    syllabus_context : str or None
        The generated syllabus content.  The agent extracts topics, modules,
        and learning objectives from this context to design relevant theory
        artifacts.  When omitted the agent infers topics from the course name.
    run_id : str or None
        The unique run identifier (e.g. ``"2026-08-23_120000_Course_Name"``).
        When provided, it is injected into the task description so the agent
        can construct the correct ``base_path`` for ``write-directory-tree``
        tool calls.
    tier : str or None
        When set to a specific tier directory name (e.g. ``"tier1_foundations"``),
        the task generates theory for that ONE tier only.  Used during
        ``--resume-from`` to regenerate individual tiers.  When ``None`` (the
        default), generates artifacts for all 3 tiers in one call
        (backward-compatible behavior).
    verbose : bool
        Enable detailed task execution logging.

    Returns
    -------
    Task
        Fully-configured CrewAI Task ready to be assigned to a Crew.
    """
    # ── Tier-specific labels ──────────────────────────────────────────
    tier_labels: dict[str, str] = {
        "tier1_foundations": "Tier 1 — Foundations",
        "tier2_application": "Tier 2 — Application",
        "tier3_architecture": "Tier 3 — Architecture",
    }
    tier_label: str | None = tier_labels.get(tier) if tier else None

    # ── Per-tier format guide (narrowed when tier is set) ─────────────
    tier_format_map: dict[str, str] = {
        "tier1_foundations": (
            "| Tier 1 — Foundations | Syntax, control flow, data structures, "
            "basic OOP, algorithms | **Format A (HTML/JS)** — visualise sorting, "
            "searching, recursion, or state transitions |"
        ),
        "tier2_application": (
            "| Tier 2 — Application | APIs, data pipelines, CLI tools, testing, "
            "databases | **Format B (Terminal Script)** — step-by-step "
            "walkthrough of API calls, ETL flows, or CLI workflows |"
        ),
        "tier3_architecture": (
            "| Tier 3 — Architecture | Microservices, Docker, CI/CD, system "
            "design, observability | **Format C (Mermaid.js Markdown)** — "
            "sequence diagrams, class diagrams, deployment topologies |"
        ),
    }

    if tier and tier in tier_format_map:
        format_guide: str = (
            "## 🎨  Format Selection Guide\n\n"
            f"For **{tier_label}**, use the recommended format below.  If the tier's "
            "concept clearly fits a different format better, use your judgment:\n\n"
            "| Tier | Typical Concepts | Recommended Format |\n"
            "|------|-----------------|--------------------|\n"
            f"{tier_format_map[tier]}\n\n"
            "**Override rule:** If this tier's concept clearly fits a different "
            "format better, use your judgment.  The table is a guideline, "
            "not a straitjacket.\n"
        )
    else:
        format_guide = _FORMAT_SELECTION_GUIDE

    # ── Cross-tier coherence guardrail (only for per-tier calls) ──────
    coherence_guardrail: str = (
        "## 🔗 Cross-Tier Coherence (Per-Tier Mode)\n\n"
        "Your theory artifact is for **{label} only**.  However, it should be "
        "understandable in isolation.  When referencing concepts from earlier "
        "tiers, add a brief 1-2 sentence inline explanation.  For example:\n\n"
        '> "In Tier 1, you learned that vectors represent direction and '
        'magnitude. Now in Tier 2, we use vectors to calculate..."\n\n'
        "Do NOT assume the student has completed earlier tiers.  If this is "
        "Tier 1, no cross-references are needed.\n"
    )
    # ---- Build the description ------------------------------------------
    description_parts: list[str] = [
        f"**🌐 LANGUAGE: ALL output (theory artifacts, README files, inline text, "
        f"UI labels in HTML, terminal script comments, Mermaid diagram labels) MUST "
        f"be written in {material_language}.**\n\n"
        f"Generate interactive theory artifacts for the following course:\n\n"
        f"**Course Name:** {course_name}\n",
    ]

    if run_id:
        description_parts.append(
            f"\n**Run ID:** {run_id}\n"
            f"(MUST use this ``run_id`` to construct the ``base_path`` for "
            f"every `write-directory-tree` tool call)\n"
        )

    if syllabus_context:
        # Truncate syllabus to ~3000 chars to keep prompt manageable.
        ctx = syllabus_context
        if len(ctx) > 4000:
            ctx = ctx[:4000] + "\n\n[... syllabus truncated for length ...]\n"
        description_parts.append(f"\n**Syllabus Context:**\n\n{ctx}\n")

    # ── Task scope: all tiers or just one? ─────────────────────────────
    if tier and tier_label:
        description_parts.append(
            f"\n\nYour task is to read the syllabus above, identify the core "
            f"concept for **{tier_label}**, and generate exactly ONE theory "
            f"artifact for this tier using the format most appropriate for "
            f"the concept.\n\n"
            f"{coherence_guardrail.format(label=tier_label)}\n\n"
            f"{format_guide}\n\n"
        )
    else:
        description_parts.append(
            f"\n\nYour task is to read the syllabus above, identify the core "
            f"concept for each of the 3 tiers, and generate exactly ONE theory "
            f"artifact per tier using the format most appropriate for the "
            f"concept.\n\n"
            f"{format_guide}\n\n"
        )

    description_parts.append(f"{_ARTIFACT_REQUIREMENTS}\n\n{_TOOL_USAGE_MANDATE}\n")

    description = "".join(description_parts)

    # ---- Build the expected_output --------------------------------------
    if tier:
        expected_output = (
            "## 🔴 CRITICAL: You MUST use the `output_export_tool`\n\n"
            'Use the `output_export_tool` with `command="write-directory-tree"` '
            "to write your theory artifact to disk.  Write exactly ONE artifact "
            f"for **{tier_label}** into the `theory/` subfolder:\n\n"
            f"- `output/<run_id>/labs/{tier}/theory/`\n\n"
            "**Once the file is written**, produce a Markdown summary listing "
            "the tier, the format chosen (A/B/C), the filename, and a one-line "
            "description of the artifact.\n"
        )
    else:
        expected_output = (
            "## 🔴 CRITICAL: You MUST use the `output_export_tool`\n\n"
            'Use the `output_export_tool` with `command="write-directory-tree"` '
            "to write ALL theory artifacts to disk.  Write exactly ONE artifact "
            "per tier into the `theory/` subfolder of each tier's lab directory:\n\n"
            "- `output/<run_id>/labs/tier1_foundations/theory/`\n"
            "- `output/<run_id>/labs/tier2_application/theory/`\n"
            "- `output/<run_id>/labs/tier3_architecture/theory/`\n\n"
            "**Once all files are written**, produce a Markdown summary listing "
            "each tier, the format chosen (A/B/C), the filename, and a one-line "
            "description of the artifact.\n"
        )

    # ---- Compute the output file path -----------------------------------
    if run_id and tier:
        output_file: str = f"output/{run_id}/labs/{tier}/theory/README.md"
    elif run_id:
        output_file: str = f"output/{run_id}/theory/README.md"
    else:
        output_file: str = "output/theory/README.md"

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
create_theory_generation_task = create_theory_task

# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.agents.theory_instructor import create_theory_instructor

    agent = create_theory_instructor(verbose=True)
    task = create_theory_task(
        course_name="Javascript OOP",
        agent=agent,
        run_id="2026-08-23_120000_Javascript_OOP",
        verbose=True,
    )
    print("✅ Theory Generation Task created successfully.\n")
    print(f"   Output file:  {task.output_file}")
    print(f"   Agent role:   {agent.role.split(chr(10))[0]}")
    print(f"   Async:        {task.async_execution}")
    print(f"   Description length: {len(task.description)} chars")
    print(f"   Expected output length: {len(task.expected_output)} chars")
