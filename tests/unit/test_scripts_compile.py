"""Unit: every scripts/*.py is syntactically valid Python."""
import pytest

from _pluginmeta import ROOT

PY_FILES = sorted(ROOT.glob("scripts/**/*.py"))


@pytest.mark.parametrize("path", PY_FILES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_script_compiles(path):
    try:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    except SyntaxError as e:
        pytest.fail(f"{path.relative_to(ROOT)}: line {e.lineno}: {e.msg}")
