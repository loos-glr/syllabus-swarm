"""
qa_review.py — QA Review Task (Technical + Didactic + Theory Checks)
====================================================================

Issue #7: AI-Driven Feedback Loop — QA Reviewer Agent

Defines a CrewAI **Task** that, when executed by the QA Reviewer agent,
performs a thorough review of all generated lab AND theory files.  The
task mandates three distinct checks:

  1. **Technical Correctness Check** — Zero syntax errors, no missing
     imports, no hallucinated variables, completely self-contained execution.
  2. **Didactic & Clarity Check** — README instructions and code comments
     use clear, accessible language suited for MBO4 vocational students.
     TODO markers must be actionable and unambiguous.
  3. **Theory Artifact Review** — Interactive HTML/JS files, terminal
     scripts, and Mermaid.js Markdown diagrams are functional, self-contained,
     and didactically appropriate for MBO4 students.

If any check fails, the QA Reviewer is explicitly instructed to use
CrewAI's delegation mechanism to assign a fix task back to the
responsible agent (Lab Developer for lab files, Theory Instructor for
theory files) with specific, actionable feedback.
"""

from __future__ import annotations

from crewai import Agent, Task


def create_qa_review_task(
    *,
    agent: Agent,
    course_name: str,
    run_id: str | None = None,
    lab_developer_role: str | None = None,
    theory_instructor_role: str | None = None,
    material_language: str = "Dutch",
    verbose: bool = False,
) -> Task:
    """Create a CrewAI Task that performs QA review of generated lab and theory files.

    Parameters
    ----------
    agent : Agent
        The QA Reviewer agent that will execute this task.
    course_name : str
        The name of the vocational course (e.g. "Full-Stack Web Development").
    run_id : str or None
        The unique run identifier (e.g. ``"2026-08-23_120000_Course_Name"``).
        Used to locate the correct per-run output directory.
    lab_developer_role : str or None
        The exact ``role`` string of the Lab Developer agent.  When provided,
        it is injected verbatim into the delegation instructions so the QA
        Reviewer can pass the correct coworker name to CrewAI's delegation
        tool (which matches against the full sanitised role, not a short
        display name).
    theory_instructor_role : str or None
        The exact ``role`` string of the Theory Instructor agent (same
        rationale as ``lab_developer_role``).
    verbose : bool
        Enable detailed task execution logging.

    Returns
    -------
    Task
        Fully-configured CrewAI Task ready to be assigned to a Crew.
    """
    # ── Determine the labs directory path ──────────────────────────────
    # Labs are written to output/<run_id>/labs/<tier>/ (no course-name
    # subdirectory), so the review root is the labs/ directory itself.
    if run_id:
        labs_dir = f"output/{run_id}/labs/"
    else:
        labs_dir = "output/labs/"

    # ── Build the description ──────────────────────────────────────────
    description = (
        f"**🌐 LANGUAGE: ALL output (QA report, findings, delegation instructions) "
        f"MUST be written in {material_language}.**\n\n"
        f"## QA Review — Generated Labs & Theory for: {course_name}\n\n"
        f"**Labs Directory:** `{labs_dir}`\n\n"
        f"---\n\n"
        f"### Your Task\n\n"
        f"You must review ALL files in the labs directory above — including "
        f"both lab code AND theory artifacts.  Use your `DirectoryReadTool` "
        f"to list the directory contents, then use your `FileReadTool` to "
        f"read every source file, README, and theory artifact.\n\n"
        f"The labs are organised into three tiers:\n"
        f"- `tier1_foundations/` — Single-file exercises with TODO markers.\n"
        f"- `tier2_application/` — Multi-file mini-projects.\n"
        f"- `tier3_architecture/` — Service-oriented systems.\n\n"
        f"Each tier has THREE subdirectories:\n"
        f"- `starter/` — Scaffolded exercises with TODO markers.\n"
        f"- `solution/` — Reference implementations.\n"
        f"- `theory/` — Interactive theory artifacts (HTML, shell scripts, "
        f"or Mermaid.js Markdown) that teach concepts BEFORE the lab work.\n\n"
        f"---\n\n"
        f"### 🔍  Check 1: Technical Correctness (Lab Code)\n\n"
        f"For EVERY source file in starter/ and solution/, verify:\n\n"
        f"- **Zero syntax errors** — The code must be valid and parseable "
        f"in the target language.  Check for missing brackets, unmatched "
        f"quotes, incorrect indentation, and invalid keywords.\n"
        f"- **No missing imports** — Every function, class, or module used "
        f"in the code must be imported or defined.  Check that all "
        f"`import`/`require`/`use` statements are present and correct.\n"
        f"- **No hallucinated variables or functions** — Every variable "
        f"referenced must be declared.  Every function called must exist "
        f"(either defined in the file, imported, or part of the standard "
        f"library).  Watch for typos in variable/function names.\n"
        f"- **Completely self-contained execution** — After following the "
        f"README instructions, a student must be able to run the code "
        f"without any additional fixes.  Check that file paths in "
        f"instructions match the actual file names.  Check that dependency "
        f"files (requirements.txt, package.json, etc.) list everything "
        f"needed.\n"
        f"- **Solution code must actually work** — The solution/ files "
        f"must be complete, runnable implementations that produce the "
        f"expected output described in the README.\n\n"
        f"### 🔍  Check 2: Didactic & Clarity Check (Lab Code)\n\n"
        f"For EVERY README.md and code comment in starter/ and solution/, "
        f"verify:\n\n"
        f"- **Clear, accessible language** — The text must be understandable "
        f"by MBO4 vocational students (Dutch HBO-equivalent, ages 16-20).  "
        f"Avoid overly academic jargon, complex theoretical explanations, "
        f"and unnecessarily long sentences.  Use concrete examples.\n"
        f"- **Actionable TODO markers** — Every `# TODO:` or `// TODO:` "
        f"marker must be a concrete, specific instruction.  A student "
        f"should know exactly *what* to implement and *how* to verify "
        f"their work.  Vague TODOs like 'implement this function' or "
        f"'add error handling' are UNACCEPTABLE.  Good examples:\n"
        f"  - `# TODO(t1): Write a function called 'calculate_average' "
        f"that takes a list of numbers and returns their mean.  Test it "
        f"with [1, 2, 3] — it should return 2.0.`\n"
        f"  - `# TODO(t2): Add a try/except block around the file.read() "
        f"call on line 42 to handle FileNotFoundError.  Print a friendly "
        f"message to the user.`\n"
        f"- **Practical, hands-on focus** — Instructions should emphasise "
        f"*doing* over *reading*.  Every concept should be immediately "
        f"followed by a code exercise.  Theory should be minimal and "
        f"directly relevant to the task at hand.\n"
        f"- **Complete README sections** — Every README must have all "
        f"required sections: Learning Objectives, Key Concepts, "
        f"Prerequisites, Getting Started, Project Structure, Lab "
        f"Instructions, Humanics Reflection, and Additional Resources.\n\n"
        f"### 🔍  Check 3: Theory Artifact Review (theory/)\n\n"
        f"For EVERY file in each tier's `theory/` subdirectory, verify:\n\n"
        f"**For HTML/JS files (Format A):**\n"
        f"- Opens in a browser without JavaScript errors — check that all "
        f"DOM queries reference elements that actually exist in the HTML.\n"
        f"- Interactive controls (buttons, sliders, inputs) are wired to "
        f"event handlers that exist in the script.\n"
        f"- Learning objectives are listed at the top of the page.\n"
        f"- The visualisation or interactive element actually demonstrates "
        f"the concept from the syllabus — not just a generic demo.\n"
        f"- Scripts are placed at the end of `<body>` or wrapped in "
        f"`DOMContentLoaded`.\n"
        f"- Error handling is present: try/catch around initialization, "
        f"null checks on DOM queries, visible error display if something "
        f"goes wrong.\n"
        f"- No broken references: SVG element IDs, data structure keys, "
        f"and function names are consistent throughout.\n\n"
        f"**For shell scripts (Format B):**\n"
        f"- Starts with `#!/usr/bin/env bash` and `set -euo pipefail`.\n"
        f"- Every step prints a clear heading explaining *what* and *why*.\n"
        f"- Pauses between steps with `read -r -p 'Press Enter to "
        f"continue...'` so the student controls the pace.\n"
        f"- Commands are correct for the target environment (check for "
        f"typos in command names, missing flags, incorrect paths).\n"
        f"- Includes a summary at the end recapping what was learned.\n\n"
        f"**For Mermaid.js Markdown files (Format C):**\n"
        f"- At least 2 distinct Mermaid diagrams are present.\n"
        f"- All ` ```mermaid ` code blocks are properly closed with ` ``` `.\n"
        f"- Diagram syntax is valid Mermaid.js that renders correctly.\n"
        f"- Explanatory prose between diagrams — not just a wall of "
        f"diagrams.\n"
        f"- Has an H1 title, learning objectives, and a 'Key Takeaways' "
        f"section.\n\n"
        f"**For ALL theory formats:**\n"
        f"- The file is truly self-contained — no broken CDN links, no "
        f"missing assets, no external dependencies that aren't documented.\n"
        f"- Language is MBO4-appropriate: clear, concrete, avoids "
        f"unnecessary academic jargon.\n"
        f"- The concept taught matches the tier's topic in the syllabus.\n"
        f"- The artifact actually *works* — a student opening/running it "
        f"should immediately see the intended interactive experience, not "
        f"a blank page or error message.\n\n"
        f"---\n\n"
        f"### 🎨  Check 4: GLR Design System Audit\n\n"
        f"Verify that ALL generated materials follow the GLR Media Creative "
        f"brand guidelines for Grafisch Lyceum Rotterdam:\n\n"
        f"**For presentations (Marp Markdown):**\n"
        f"- [ ] NOT using generic `theme: uncover` — must use GLR custom theme\n"
        f"- [ ] Colours: `#76B800`/`#A6E22E` (green), `#000000` (black), "
        f"`#002BFF` (blue accent)\n"
        f"- [ ] 0px border-radius on all elements (no rounded corners)\n"
        f"- [ ] Space Grotesk for headings, Hanken Grotesk for body\n"
        f"- [ ] GLR component classes used where applicable "
        f"(`.glr-card`, `.glr-badge`, `.glr-btn-primary`, etc.)\n\n"
        f"**For HTML theory artifacts:**\n"
        f"- [ ] Uses GLR colour palette (green `#76B800`, black `#000000`, "
        f"white `#FFFFFF`)\n"
        f"- [ ] 0px border-radius on all elements\n"
        f"- [ ] Hard offset shadows only (no blur/glow)\n"
        f"- [ ] Space Grotesk headings, Hanken Grotesk body\n"
        f"- [ ] Google Fonts import includes both families\n\n"
        f"**For all documents:**\n"
        f"- [ ] Clean, professional formatting aligned with High-Contrast "
        f"Brutalist Modernism\n"
        f"- [ ] Humanics tags `[T]` `[D]` `[H]` present and correctly styled\n"
        f"- [ ] Brand footer/credit where appropriate: "
        f"\"Generated for Grafisch Lyceum Rotterdam (GLR)\"\n\n"
        f"---\n\n"
        f"### 🔴  CRITICAL: Delegation Mandate\n\n"
        f"If ANY file fails ANY check, you **MUST** use CrewAI's "
        f"delegation mechanism (`delegate_work_to_coworker`) to assign a "
        f"fix task back to the responsible agent:\n\n"
        f"- **Lab code issues (starter/ or solution/)** → Delegate to the "
        f"Lab Developer.  The EXACT coworker name you must use is:\n"
        f"  **`{lab_developer_role or 'lab & project developer'}`**\n"
        f"- **Theory artifact issues (theory/)** → Delegate to the "
        f"Theory Instructor.  The EXACT coworker name you must use is:\n"
        f"  **`{theory_instructor_role or 'theory instructor'}`**\n\n"
        f"⚠️  The coworker name MUST exactly match the role string shown "
        f"above — CrewAI matches by doing case‑insensitive whitespace‑"
        f"normalised comparison of the FULL role, so a short name like "
        f'"Lab Developer" alone will NOT match.\n\n'
        f"Your delegation message must include:\n"
        f"1. **Which specific files** need to be fixed (full paths).\n"
        f"2. **What exactly is wrong** in each file (be specific — quote "
        f"the problematic code or text).\n"
        f"3. **Why it matters** for MBO4 students (explain the impact on "
        f"learning).\n"
        f"4. **How to fix it** (provide concrete guidance, not just "
        f"'fix this').\n\n"
        f"### 🔴  CRITICAL: Delegated Agents MUST Write to the Correct Directory\n\n"
        f"When you delegate a fix, you **MUST** instruct the coworker to use "
        f"the **exact** `run_id` shown in the Labs Directory path above "
        f'(e.g., `run_id="{run_id or "RUN_ID"}"`).  The coworker MUST use this '
        f"`run_id` in ALL `write-labs` and `write-directory-tree` tool calls.\n\n"
        f"❌ **WRONG:** using a made-up run_id like "
        f'`run_id="2023-06-15_120000_Some_Random_Name"`\n'
        f'✅ **CORRECT:** `run_id="{run_id or "RUN_ID"}"` '
        f"(the run_id from the Labs Directory above)\n\n"
        f"Using the wrong `run_id` causes files to be written to a completely "
        f"different directory, making them invisible to this review. "
        f"This is the #1 cause of 'I fixed it but the fix didn't appear.'\n\n"
        f"Do NOT attempt to fix the files yourself.  Your role is to "
        f"*review and delegate*, not to *rewrite*.  The Lab Developer "
        f"and Theory Instructor are responsible for implementing fixes "
        f"based on your feedback.\n\n"
        f"---\n\n"
        f"### 📋  Expected Output\n\n"
        f"Produce a comprehensive **Markdown QA Report** with the following "
        f"sections:\n\n"
        f"```markdown\n"
        f"# QA Review Report — {course_name}\n\n"
        f"## Summary\n"
        f"- Total files reviewed: [count]\n"
        f"- Lab technical issues found: [count]\n"
        f"- Lab didactic issues found: [count]\n"
        f"- Theory artifact issues found: [count]\n"
        f"- Delegation(s) sent: [Lab Developer: yes/no, Theory Instructor: yes/no]\n"
        f"- Final verdict: [PASSED / NEEDS FIXES]\n\n"
        f"## Technical Correctness Check (Lab Code)\n"
        f"[For each lab file: PASSED or FAILED with specific issues]\n\n"
        f"## Didactic & Clarity Check (Lab Code)\n"
        f"[For each lab file: PASSED or FAILED with specific issues]\n\n"
        f"## Theory Artifact Review\n"
        f"[For each theory file: PASSED or FAILED with specific issues]\n\n"
        f"## Delegation Summary\n"
        f"[If fixes were delegated: what was sent to which agent]\n\n"
        f"## Sign-Off\n"
        f"[Final confirmation that ALL content — labs AND theory — is "
        f"ready for students, or a clear statement that fixes are still "
        f"needed]\n"
        f"```\n\n"
        f"If ALL checks pass for ALL files (labs AND theory), your sign-off "
        f"must explicitly state: '✅ **QA SIGN-OFF: All labs and theory "
        f"artifacts are technically correct and didactically appropriate "
        f"for MBO4 students.  Ready for classroom use.**'\n"
    )

    expected_output = (
        "A comprehensive Markdown QA Report with Technical Correctness "
        "Check results for lab code, Didactic & Clarity Check results for "
        "lab code, Theory Artifact Review results for theory/ files, a "
        "Delegation Summary (if any fixes were delegated to the Lab "
        "Developer or Theory Instructor), and a final Sign-Off confirming "
        "readiness for MBO4 students."
    )

    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
        async_execution=False,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.agents.qa_reviewer import create_qa_reviewer

    agent = create_qa_reviewer(verbose=True)
    task = create_qa_review_task(
        agent=agent,
        course_name="JavaScript OOP Basics",
        run_id="2026-08-23_120000_JavaScript_OOP_Basics",
        verbose=True,
    )
    print("✅ QA Review Task created successfully.\n")
    print(f"   Agent role:   {agent.role.split(chr(10))[0]}")
    print(f"   Async:        {task.async_execution}")
    print(f"   Description length: {len(task.description)} chars")
    print(f"   Expected output length: {len(task.expected_output)} chars")
