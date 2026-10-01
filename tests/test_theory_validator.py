"""
test_theory_validator.py — Tests for src/exporters/theory_validator.py
======================================================================

Covers the deterministic HTML checks in the post-generation theory
validator, with a focus on the fixes that were added together:

  • Regression — the null-check heuristic must accept combined guards
    (``if (!a || !b)``) and truthy guards (``if (el)``); it used to emit
    false-positive "used without a null check" warnings.
  • Regression — a ``<script>`` that is the LAST element of ``<body>`` must
    not be flagged as "may run before elements exist".
  • New checks — absolute positioning + negative margin (label overlap),
    dead BEM selectors, low-contrast text colours, and ``keydown``
    listeners that do not skip form controls.

All fixtures are written to ``tmp_path`` so no real files are touched.
"""

from __future__ import annotations

from pathlib import Path

from src.exporters.theory_validator import validate_theory_file

# ---------------------------------------------------------------------------
# Fixtures (HTML snippets)
# ---------------------------------------------------------------------------

GOOD_HTML = """<!DOCTYPE html>
<html lang="nl">
<head>
<meta charset="UTF-8">
<title>Good</title>
<style>
.bar { margin-bottom: 2.75rem; }
.progress-label {
    position: absolute;
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    width: max-content;
    white-space: nowrap;
    color: #555;
}
</style>
</head>
<body>
<div id="a"></div>
<div id="b"></div>
<button id="go">Go</button>
<script>
var a = document.getElementById('a');
var b = document.getElementById('b');
var btn = document.getElementById('go');
if (!a || !b) { console.error('missing elements'); }
if (btn) { btn.addEventListener('click', function () {}); }
try { console.log('ok'); } catch (err) { console.error(err); }
document.addEventListener('keydown', function (e) {
    var tag = (e.target && e.target.tagName) || '';
    if (tag === 'SELECT' || tag === 'INPUT' || tag === 'TEXTAREA') { return; }
});
</script>
</body>
</html>
"""

BAD_HTML = """<!DOCTYPE html>
<html lang="nl">
<head>
<meta charset="UTF-8">
<title>Bad</title>
<style>
.progress-label { position: absolute; left: 50%; margin-left: -1.25rem; }
.toggle button--active { background: #000; }
.hint { color: #999; }
</style>
</head>
<body>
<div id="app"></div>
<script>
document.getElementById('app').textContent = 'hi';
document.addEventListener('keydown', function (e) {
    if (e.key === 'ArrowRight') { console.log('right'); }
});
</script>
</body>
</html>
"""

UNCHECKED_HTML = """<!DOCTYPE html>
<html lang="nl">
<head><meta charset="UTF-8"><title>Unchecked</title></head>
<body>
<div id="y"></div>
<script>
var y = document.getElementById('y');
y.focus();
</script>
</body>
</html>
"""


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def _messages(path: Path) -> list[str]:
    return [issue.message for issue in validate_theory_file(path).issues]


# ---------------------------------------------------------------------------
# Regression — no false positives
# ---------------------------------------------------------------------------


class TestNoFalsePositives:
    """The validator must not flag correct, defensive HTML/JS."""

    def test_combined_and_truthy_null_checks_are_accepted(self, tmp_path: Path) -> None:
        """``if (!a || !b)`` and ``if (btn)`` count as null checks."""
        joined = " ".join(_messages(_write(tmp_path, "good.html", GOOD_HTML))).lower()
        assert "null check" not in joined

    def test_last_script_in_body_is_not_flagged(self, tmp_path: Path) -> None:
        """A trailing <script> must not be reported as running too early."""
        joined = " ".join(_messages(_write(tmp_path, "good.html", GOOD_HTML))).lower()
        assert "may run before elements exist" not in joined

    def test_centred_absolute_label_is_not_flagged(self, tmp_path: Path) -> None:
        """``left: 50%`` + ``translateX(-50%)`` is the CORRECT pattern."""
        joined = " ".join(_messages(_write(tmp_path, "good.html", GOOD_HTML))).lower()
        assert "negative 'margin-left'" not in joined

    def test_guarded_keydown_is_not_flagged(self, tmp_path: Path) -> None:
        """A keydown listener that skips form controls is fine."""
        joined = " ".join(_messages(_write(tmp_path, "good.html", GOOD_HTML))).lower()
        assert "keydown" not in joined

    def test_good_html_has_no_new_css_warnings(self, tmp_path: Path) -> None:
        """No BEM/contrast warnings for clean CSS."""
        joined = " ".join(_messages(_write(tmp_path, "good.html", GOOD_HTML))).lower()
        assert "bem modifier" not in joined
        assert "low-contrast" not in joined


# ---------------------------------------------------------------------------
# New checks — must trigger on bad HTML
# ---------------------------------------------------------------------------


class TestNewCssAndA11yChecks:
    """The new heuristics should fire on the classic defects."""

    def test_absolute_plus_negative_margin_is_flagged(self, tmp_path: Path) -> None:
        joined = " ".join(_messages(_write(tmp_path, "bad.html", BAD_HTML)))
        assert "negative 'margin-left'" in joined

    def test_dead_bem_selector_is_flagged(self, tmp_path: Path) -> None:
        joined = " ".join(_messages(_write(tmp_path, "bad.html", BAD_HTML)))
        assert "BEM modifier" in joined

    def test_low_contrast_text_colour_is_flagged(self, tmp_path: Path) -> None:
        joined = " ".join(_messages(_write(tmp_path, "bad.html", BAD_HTML)))
        assert "Low-contrast text colour '#999'" in joined

    def test_unguarded_keydown_is_flagged(self, tmp_path: Path) -> None:
        joined = " ".join(_messages(_write(tmp_path, "bad.html", BAD_HTML)))
        assert "'keydown' listener" in joined


# ---------------------------------------------------------------------------
# The null-check heuristic still catches genuinely unchecked queries
# ---------------------------------------------------------------------------


class TestNullCheckStillCatchesMissingGuard:
    """No false negatives: an unchecked DOM variable must still warn."""

    def test_unchecked_dom_variable_is_flagged(self, tmp_path: Path) -> None:
        joined = " ".join(_messages(_write(tmp_path, "unchecked.html", UNCHECKED_HTML)))
        assert "used without a null check" in joined
