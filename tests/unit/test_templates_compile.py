"""Unit: every *.py.template renders to syntactically valid Python.

Mustache-style conditional sections (`{{#X}}…{{/X}}`, `{{^X}}…`) are stripped to
their all-included variant, then value placeholders (`{{VAR}}`) become a valid
identifier — so we check the template's actual code structure (syntax only, not
name binding or which conditional branch renders).
"""
import re

import pytest

from _pluginmeta import ROOT

TEMPLATES = sorted(ROOT.glob("knowledge/templates/**/*.py.template"))
_SECTION = re.compile(r"\{\{[#/^][^}]*\}\}")   # {{#X}} {{/X}} {{^X}} section markers
_PLACEHOLDER = re.compile(r"\{\{[^}]*\}\}")     # {{VAR}} value placeholders


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_template_compiles(path):
    src = _SECTION.sub("", path.read_text(encoding="utf-8"))
    src = _PLACEHOLDER.sub("__PH__", src)
    try:
        compile(src, str(path), "exec")
    except SyntaxError as e:
        pytest.fail(
            f"{path.relative_to(ROOT)}: syntax error after placeholder substitution "
            f"(line {e.lineno}): {e.msg}"
        )
