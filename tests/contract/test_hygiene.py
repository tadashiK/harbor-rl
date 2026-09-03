"""Contract: open-source hygiene — no personal abs paths, no identity leaks.

Permanently guards the cleanups done before open-sourcing (personal home paths,
the pql / ddiffpg / bidex / W&B-handle references). tests/ itself is not scanned
(this file legitimately contains the identity tokens in its regex).

`supersglzc` stays on the denylist because it must not reappear as a stray home
path or W&B handle — but the plugin's own marketplace URL is exempted: a
published plugin has to name the repo it is installed from.

`pql` carries the same shape of exemption, for the same reason and one more. The
custom_torch tree is adapted from PQL, which is MIT, and an MIT notice that cannot
name the project it covers is not a notice — scrubbing the token is precisely what
left that tree unattributed. So the attribution header rendered into each derived
template, THIRD_PARTY_NOTICES.md, and the README licence paragraph may name it; a
bare `pql` anywhere else is still a leak and still fails.
"""
import re

import pytest

from _pluginmeta import ROOT, iter_text_files

FILES = sorted(iter_text_files(), key=lambda p: p.as_posix())

_ABS_RE = re.compile(r"/(?:home|Users)/([^/\s`\"')]+)")
_ALLOWED_USERS = {"me", "you", "user", "USER"}
_IDENTITY_RE = re.compile(
    r"(?<![A-Za-z])(?:pql|ddiffpg|supersglzc\w*)(?![A-Za-z])"
    r"|(?<![A-Za-z])bidex(?!hands)(?![A-Za-z])",
    re.IGNORECASE,
)


# The plugin's own published URLs — marketplace / clone, and the GitHub Pages host the
# README and docs site link to. These are the only places the publisher handle is
# expected. Stripped before the identity scan so any OTHER use of the handle (home
# paths, W&B entities) still fails.
_PUBLISHER_URL_RE = re.compile(
    r"(?:github\.com/)?supersglzc/harbor(?:-dev)?(?:\.git)?"
    r"|supersglzc\.github\.io"
)

# The upstream-attribution exemptions. Deliberately narrow: the exact header rendered into
# every derived template, and the one file whose job is to collect upstream notices. Both
# name a PROJECT, not a handle — which is the distinction the denylist is drawing.
_ATTRIBUTION_RE = re.compile(
    r"# Adapted from PQL \(https://github\.com/Improbable-AI/pql\)"
)
# The files where naming an upstream project is the POINT: the notices file itself, and the
# README's licence paragraph. Only the project-name token is stripped in them — the personal
# abs-path check and every other identity token still run over these files unchanged.
_ATTRIBUTION_FILES = {"THIRD_PARTY_NOTICES.md", "README.md"}
_UPSTREAM_NAME_RE = re.compile(r"PQL|Improbable-AI/pql", re.IGNORECASE)


def _is_placeholder_user(u):
    return u.startswith("<") or set(u) <= {"."}


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_no_personal_abs_paths(path):
    bad = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        for user in _ABS_RE.findall(line):
            if user in _ALLOWED_USERS or _is_placeholder_user(user):
                continue
            bad.append(line.strip()[:120])
    assert not bad, f"{path.relative_to(ROOT)}: personal absolute path(s): {bad[:3]}"


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_no_identity_leaks(path):
    rel = path.relative_to(ROOT).as_posix()
    text = _PUBLISHER_URL_RE.sub("", path.read_text(encoding="utf-8", errors="ignore"))
    text = _ATTRIBUTION_RE.sub("", text)
    if rel in _ATTRIBUTION_FILES:
        text = _UPSTREAM_NAME_RE.sub("", text)
    hits = {m.group(0) for m in _IDENTITY_RE.finditer(text)}
    assert not hits, f"{rel}: identity token(s): {sorted(hits)}"


def test_published_repo_name_is_consistent():
    """Every place that names the repo must name the SAME repo.

    The manifest said `supersglzc/harbor` while the repository is `harbor-rl`, and the
    bundled installer printed `/plugin marketplace add supersglzc/harbor` — an instruction
    that resolves to nothing. The denylist above could not catch it: it allows both spellings,
    because both are the publisher's handle. What is wrong is the disagreement, not the name.
    """
    import json
    import re

    expected = "supersglzc/harbor-rl"
    wrong = []

    manifest = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text())
    for key in ("homepage", "repository"):
        if expected not in manifest.get(key, ""):
            wrong.append(f".claude-plugin/plugin.json:{key} = {manifest.get(key)!r}")

    # Anywhere the handle appears with a repo path, it must be the real one.
    for rel in ("README.md", "docs/guide/install.md", "CITATION.cff",
                "scripts/install/install_prerequisites.sh"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        for m in re.findall(r"supersglzc/harbor[A-Za-z0-9._-]*", text):
            if m.rstrip(".git").rstrip("/") != expected:
                wrong.append(f"{rel}: {m}")

    assert not wrong, "repo name disagrees with " + expected + ":\n  " + "\n  ".join(wrong)
