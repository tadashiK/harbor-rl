#!/usr/bin/env python3
"""dependency-generator smoke runner.

Runs against a host-side venv at `<repo>/.venv`. Intended to be called as
Step 6 of the dependency-generator agent.

Phases (controlled by CLI flags):
  --check-prereqs   verify uv + python3 are on PATH (host)
  --build           run `bash <repo>/harbor/dependency-generator/setup_uv.sh` to create + populate .venv
  (always)          Tier 1 (nvidia-smi + torch.cuda) and Tier 2 (project imports)
                    via `<repo>/.venv/bin/python -c "..."`.

Outputs JSON the agent can copy `smoke` verbatim into its return JSON.

Usage:
    python smoke_uv.py <repo> [--check-prereqs] [--build]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path | None = None, timeout: int = 600) -> tuple[int, str, str]:
    try:
        cp = subprocess.run(
            cmd, cwd=str(cwd) if cwd else None,
            capture_output=True, text=True, timeout=timeout,
        )
        return cp.returncode, cp.stdout, cp.stderr
    except subprocess.TimeoutExpired as e:
        return 124, e.stdout or "", e.stderr or ""


def _venv_python(repo: Path) -> Path:
    return repo / ".venv" / "bin" / "python"


def check_prereqs() -> tuple[str, str]:
    if shutil.which("uv") is None:
        return ("fail", "uv not found on PATH — install via 'curl -fsSL https://astral.sh/uv/install.sh | sh'")
    if shutil.which("python3") is None:
        return ("fail", "python3 not found on PATH")
    return ("pass", "")


def build(repo: Path) -> tuple[str, str]:
    setup = repo / "harbor" / "dependency-generator" / "setup_uv.sh"
    if not setup.is_file():
        return ("fail", f"{setup} not found — run render_uv.py first")
    rc, out, err = _run(["bash", str(setup)], cwd=repo, timeout=1800)
    if rc != 0:
        tail = "\n".join((out + err).splitlines()[-50:])
        return ("fail", tail)
    return ("pass", "")


def tier1_basic_env(repo: Path) -> dict:
    py = _venv_python(repo)
    if not py.is_file():
        return {"verdict": "fail", "nvidia_smi": "fail", "torch_cuda": "fail",
                "device_count": 0, "error": f"{py} not found — venv missing"}
    rc_nvsmi, _out, _err = _run(["nvidia-smi"], timeout=30)
    nvsmi_ok = rc_nvsmi == 0

    # --- try torch first ---
    rc, out, err = _run([str(py), "-c",
                         "import torch; print(int(torch.cuda.is_available()), torch.cuda.device_count())"],
                        timeout=120)
    if rc == 0:
        parts = (out or "").strip().split()
        avail = parts[0] == "1" if parts else False
        n = int(parts[1]) if len(parts) > 1 else 0
        return {
            "verdict": "pass" if (nvsmi_ok and avail and n >= 1) else "partial" if avail else "fail",
            "nvidia_smi": "pass" if nvsmi_ok else "fail",
            "torch_cuda": "pass" if avail else "fail",
            "device_count": n,
            "error": "" if avail else "torch.cuda.is_available() returned False",
        }

    # torch failed — check whether it's simply not installed (JAX-first repo)
    is_no_module = "No module named 'torch'" in (err or "") or "No module named 'torch'" in (out or "")
    if not is_no_module:
        # torch is present but broken — treat as hard fail
        tail = (err or out).strip().splitlines()[-1] if (err or out).strip() else "torch import failed"
        return {"verdict": "fail", "nvidia_smi": "pass" if nvsmi_ok else "fail",
                "torch_cuda": "fail", "device_count": 0, "error": tail}

    # --- torch not installed: fall back to JAX GPU check ---
    rc2, out2, err2 = _run(
        [str(py), "-c",
         "import jax; devs = jax.devices('gpu'); print(len(devs))"],
        timeout=120,
    )
    if rc2 == 0:
        n = int((out2 or "0").strip().split()[0]) if (out2 or "").strip() else 0
        gpu_ok = n >= 1
        return {
            "verdict": "pass" if (nvsmi_ok and gpu_ok) else "partial" if gpu_ok else "fail",
            "nvidia_smi": "pass" if nvsmi_ok else "fail",
            "torch_cuda": "pass" if gpu_ok else "fail",  # field represents GPU framework availability
            "device_count": n,
            "error": "" if gpu_ok else "torch not installed; jax.devices('gpu') returned 0 devices",
        }

    # JAX GPU check also failed — report informatively
    jax_err = (err2 or out2 or "").strip().splitlines()[-1] if (err2 or out2 or "").strip() else ""
    return {
        "verdict": "fail" if not nvsmi_ok else "partial",
        "nvidia_smi": "pass" if nvsmi_ok else "fail",
        "torch_cuda": "fail",
        "device_count": 0,
        "error": f"torch not installed; jax GPU check failed: {jax_err}",
    }


_STDLIB_ROOTS = {
    "abc", "argparse", "ast", "asyncio", "base64", "binascii", "builtins", "bz2",
    "calendar", "cmath", "cmd", "codecs", "collections", "concurrent", "configparser",
    "contextlib", "contextvars", "copy", "copyreg", "csv", "ctypes", "dataclasses",
    "datetime", "decimal", "difflib", "dis", "doctest", "email", "encodings", "enum",
    "errno", "faulthandler", "fcntl", "filecmp", "fileinput", "fnmatch", "fractions",
    "functools", "gc", "getopt", "getpass", "gettext", "glob", "graphlib", "grp",
    "gzip", "hashlib", "heapq", "hmac", "html", "http", "imaplib", "imp",
    "importlib", "inspect", "io", "ipaddress", "itertools", "json", "keyword",
    "lib2to3", "linecache", "locale", "logging", "lzma", "mailbox", "mailcap",
    "math", "mimetypes", "mmap", "msvcrt", "multiprocessing", "netrc", "nntplib",
    "ntpath", "numbers", "operator", "optparse", "os", "pathlib", "pdb", "pickle",
    "pickletools", "pipes", "platform", "plistlib", "poplib", "posixpath",
    "pprint", "profile", "pstats", "pty", "pwd", "py_compile", "pyclbr", "pydoc",
    "queue", "quopri", "random", "re", "readline", "reprlib", "resource", "rlcompleter",
    "runpy", "sched", "secrets", "select", "selectors", "shelve", "shlex", "shutil",
    "signal", "site", "smtpd", "smtplib", "sndhdr", "socket", "socketserver",
    "sqlite3", "ssl", "stat", "statistics", "string", "stringprep", "struct",
    "subprocess", "sunau", "symtable", "sys", "sysconfig", "syslog", "tabnanny",
    "tarfile", "telnetlib", "tempfile", "termios", "test", "textwrap", "threading",
    "time", "timeit", "tkinter", "token", "tokenize", "tomllib", "trace", "traceback",
    "tracemalloc", "tty", "turtle", "types", "typing", "unicodedata", "unittest",
    "urllib", "uu", "uuid", "venv", "warnings", "wave", "weakref", "webbrowser",
    "winreg", "winsound", "wsgiref", "xdrlib", "xml", "xmlrpc", "zipapp", "zipfile",
    "zipimport", "zlib", "zoneinfo",
}

_IMPORT_RE = re.compile(r"^\s*(?:from\s+([a-zA-Z_][\w\.]*)\s+import|import\s+([a-zA-Z_][\w\.]*))",
                        re.MULTILINE)


def _collect_imports(repo: Path, import_name: str) -> list[str]:
    """Collect third-party top-level imports from the package __init__ and scripts.

    *import_name* is the importable name (underscores, not hyphens).
    """
    candidates = []
    pkg_init = repo / import_name / "__init__.py"
    if pkg_init.is_file():
        candidates.append(pkg_init)
    for entry in ("scripts/", "src/"):
        d = repo / entry
        if d.is_dir():
            for py in sorted(d.rglob("*.py"))[:5]:
                candidates.append(py)
    seen: set[str] = set()
    for fp in candidates:
        try:
            text = fp.read_text(errors="ignore")
        except Exception:
            continue
        for m in _IMPORT_RE.finditer(text):
            mod = m.group(1) or m.group(2)
            if not mod:
                continue
            top = mod.split(".")[0]
            if top in _STDLIB_ROOTS or top.startswith("_"):
                continue
            if top == import_name:
                continue
            seen.add(top)
    out = sorted(seen)
    if import_name and (repo / import_name / "__init__.py").is_file():
        out.insert(0, import_name)
    return out


def tier2_project_imports(repo: Path, project_name: str) -> dict:
    py = _venv_python(repo)
    if not py.is_file():
        return {"verdict": "skipped", "tested_files": [], "imports_total": 0,
                "imports_passed": 0, "imports_failed": []}
    import_name = _import_name(repo, project_name)
    mods = _collect_imports(repo, import_name)
    if not mods:
        return {"verdict": "skipped", "tested_files": [], "imports_total": 0,
                "imports_passed": 0, "imports_failed": []}
    src = ";".join(f"import {m}" for m in mods)
    rc, _out, err = _run([str(py), "-c", src], cwd=repo, timeout=300)
    init_path = import_name + "/__init__.py"
    if rc == 0:
        return {"verdict": "pass", "tested_files": [init_path],
                "imports_total": len(mods), "imports_passed": len(mods), "imports_failed": []}
    failures: list[dict] = []
    passed = 0
    for m in mods:
        rc1, _o1, err1 = _run([str(py), "-c", f"import {m}"], cwd=repo, timeout=120)
        if rc1 == 0:
            passed += 1
        else:
            tail = (err1 or "").strip().splitlines()[-1] if (err1 or "").strip() else "import failed"
            failures.append({"module": m, "error": tail[:200]})
    return {
        "verdict": "fail" if passed == 0 else ("partial" if failures else "pass"),
        "tested_files": [init_path] if (repo / import_name / "__init__.py").is_file() else [],
        "imports_total": len(mods), "imports_passed": passed, "imports_failed": failures,
    }


def _project_name(repo: Path) -> str:
    """Return the pyproject `name` field (may contain hyphens, e.g. 'loco-mujoco')."""
    pyproject = repo / "pyproject.toml"
    if pyproject.is_file():
        for line in pyproject.read_text().splitlines():
            line = line.strip()
            if line.startswith("name") and "=" in line:
                _, _, val = line.partition("=")
                val = val.strip().strip('"').strip("'")
                if val:
                    return val
    return repo.name


def _import_name(repo: Path, project_name: str) -> str:
    """Resolve the importable Python package name for *project_name*.

    pyproject `name` may use hyphens (e.g. 'loco-mujoco') while the actual
    package directory uses underscores ('loco_mujoco').  Try the underscore
    form first on disk; fall back to the raw name.
    """
    underscore = project_name.replace("-", "_")
    if (repo / underscore / "__init__.py").is_file():
        return underscore
    if (repo / project_name / "__init__.py").is_file():
        return project_name
    # No __init__.py found — return underscore form as best guess.
    return underscore


def overall_verdict(t1: str, t2: str) -> str:
    """Combine the two tier verdicts into the smoke's overall verdict.

    Precedence, and why: a tier1 failure means the env itself is broken (no CUDA, no torch),
    which makes tier2 meaningless — so it dominates. `partial` tier2 (some project imports
    resolve, some don't) is a real but non-blocking result. A `skipped` tier2 alongside a
    passing tier1 is still a pass: not every repo exposes an importable package.
    """
    if t1 == "fail" or t2 == "fail":
        return "fail"
    if t2 == "partial":
        return "partial"
    if t1 == "pass" and t2 in ("pass", "skipped"):
        return "pass"
    return "partial"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("repo", type=Path)
    p.add_argument("--check-prereqs", action="store_true")
    p.add_argument("--build", action="store_true")
    args = p.parse_args()
    repo = args.repo.resolve()
    if not repo.is_dir():
        print(f"[error] repo not a directory: {repo}", file=sys.stderr)
        return 2

    smoke = {
        "host_prereq":  "skipped",
        "build":        "skipped",
        "container_up": "skipped",   # always skipped in uv mode
        "tier1_basic_env": {"verdict": "skipped", "nvidia_smi": "skipped",
                            "torch_cuda": "skipped", "device_count": 0, "error": ""},
        "tier2_project_imports": {"verdict": "skipped", "tested_files": [],
                                  "imports_total": 0, "imports_passed": 0,
                                  "imports_failed": []},
        "overall": "skipped",
    }
    build_log_tail = ""

    if args.check_prereqs:
        v, err = check_prereqs()
        smoke["host_prereq"] = v
        if v == "fail":
            smoke["overall"] = "fail"
            print(json.dumps({"smoke": smoke, "build_log_tail": err}, indent=2))
            return 1

    if args.build:
        v, tail = build(repo)
        smoke["build"] = v
        if v == "fail":
            build_log_tail = tail
            smoke["overall"] = "fail"
            print(json.dumps({"smoke": smoke, "build_log_tail": build_log_tail}, indent=2))
            return 1

    smoke["tier1_basic_env"] = tier1_basic_env(repo)
    project_name = _project_name(repo)
    smoke["tier2_project_imports"] = tier2_project_imports(repo, project_name)

    smoke["overall"] = overall_verdict(smoke["tier1_basic_env"]["verdict"],
                                       smoke["tier2_project_imports"]["verdict"])

    print(json.dumps({"smoke": smoke, "build_log_tail": build_log_tail}, indent=2))
    return 0 if smoke["overall"] in ("pass", "partial") else 1


if __name__ == "__main__":
    raise SystemExit(main())
