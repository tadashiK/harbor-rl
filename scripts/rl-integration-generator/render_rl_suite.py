"""
Render the RL suite scaffold into a target benchmark repo.

Inputs:
  --repo                <abs path>        target repo (must already have .venv/
                                          and harbor/benchmark-generator/benchmark-spec.json)
  --algorithm-source    custom_torch | stable_baseline3 | local_implementation
                        (legacy aliases custom-torch / stable-baselines3 also accepted)
  --algorithm-package   for local_implementation: filesystem path OR github URL
                        (ignored for the other two sources)
  --algorithms          comma-separated subset of {ppo,sac,td3} (default: all). Ignored
                        for local_implementation (one shim trio per source).
  --tasks               comma-separated task IDs OR JSON array of task objects
  --gpu-sim             true | false  (read from benchmark-spec; CLI override allowed)
  --wandb-enabled       true | false
  --wandb-project       <project name or empty>
  --wandb-entity        <entity or empty>
  --wandb-mode          online | offline | disabled

Outputs (written into <repo>):
  harbor/scripts/rl/<algorithm_slug>/{train,eval,render,env_wrapper}.py
       + algo/, replay/, models/, utils/ subdirs (custom_torch / custom_jax only)
  harbor/configs/rl/{suite,ppo,sac,td3}.yaml + .parallel siblings   (shared across all sources)
  docker/docker-compose.rl.yaml                                       (best-effort; unused in uv mode)
  harbor/rl-integration-generator/rl-suite-spec.json
  harbor/rl_experiments/{sweeps,tunes}/
  harbor/utils/data_logger.py (only if missing)
  harbor/rl-integration-generator/rl-integration.md (Layer 6b user receipt)

Convention: all writes are idempotent (overwrite). User edits should go into
harbor/configs/rl/<algo>.local.yaml (Hydra overlays it on top of the canonical config).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = PLUGIN_ROOT / "knowledge" / "templates" / "rl-integration-generator"


# Canonical algorithm-source names. Older callers may pass dashed variants;
# we normalize on input and emit the canonical underscore form everywhere.
SOURCE_ALIASES = {
    "custom-torch": "custom_torch",
    "custom-jax": "custom_jax",
    "stable-baselines3": "stable_baseline3",
    "stable_baselines3": "stable_baseline3",
    "stable-baseline3": "stable_baseline3",
    "local-implementation": "local_implementation",
}
VALID_SOURCES = {"custom_torch", "custom_jax", "stable_baseline3", "local_implementation"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--algorithm-source", required=True)
    p.add_argument("--algorithm-package", default="",
                   help="local_implementation: path or github URL (else ignored)")
    p.add_argument("--algorithms", default="ppo,sac,td3")
    p.add_argument("--tasks", required=True,
                   help="Comma-separated task IDs OR JSON array of task objects")
    p.add_argument("--gpu-sim", default="auto",
                   help="true | false | auto (read from benchmark-spec.json)")
    p.add_argument("--wandb-enabled", default="true")
    p.add_argument("--wandb-project", default="harbor-rl")
    p.add_argument("--wandb-entity", default="")
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    return p.parse_args()


def normalize_source(raw: str) -> str:
    s = SOURCE_ALIASES.get(raw, raw)
    if s not in VALID_SOURCES:
        print(f"[error] unknown algorithm-source '{raw}'. Valid: {sorted(VALID_SOURCES)}",
              file=sys.stderr)
        sys.exit(2)
    return s


def derive_local_slug(package: str) -> str:
    """For local_implementation: derive a Python-safe directory name from path or URL."""
    pkg = package.strip()
    if not pkg:
        return "local_impl"
    # strip trailing slash, .git, query strings
    pkg = re.sub(r"[?#].*$", "", pkg)
    pkg = pkg.rstrip("/")
    if pkg.endswith(".git"):
        pkg = pkg[:-4]
    base = pkg.rsplit("/", 1)[-1]
    # lowercase + dashes → underscores; strip non-alnum_
    slug = re.sub(r"[^a-z0-9_]+", "_", base.lower()).strip("_")
    return slug or "local_impl"


def parse_tasks(raw: str) -> list[dict]:
    raw = raw.strip()
    if raw.startswith("["):
        return json.loads(raw)
    out: list[dict] = []
    for tok in [t.strip() for t in raw.split(",") if t.strip()]:
        out.append({
            "id": tok,
            "make": "gymnasium.make",
            "max_episode_steps": 1000,
            "success_metric": None,
            "reward_metric": "episode_return",
        })
    return out


def render_template(src: Path, dst: Path, mapping: dict[str, str]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    text = src.read_text()
    for k, v in mapping.items():
        text = text.replace("{{" + k + "}}", v)
    dst.write_text(text)


def read_gpu_sim(repo: Path) -> bool:
    spec_path = repo / "harbor" / "benchmark-generator" / "benchmark-spec.json"
    if not spec_path.exists():
        print(f"[warn] {spec_path} not found — defaulting gpu_sim=false", file=sys.stderr)
        return False
    try:
        spec = json.loads(spec_path.read_text())
    except json.JSONDecodeError:
        return False
    return bool(spec.get("gpu_sim", False))


def main():
    args = parse_args()
    repo: Path = args.repo.resolve()
    # dependency-generator must have produced <repo>/.venv + harbor/dependency-generator/setup_uv.sh.
    has_uv = (repo / ".venv" / "bin" / "python").exists() and (repo / "harbor" / "dependency-generator" / "setup_uv.sh").exists()
    if not has_uv:
        print(f"[error] {repo} is missing .venv or harbor/dependency-generator/setup_uv.sh — "
              "run dependency-generator + benchmark-generator first.", file=sys.stderr)
        sys.exit(2)

    source = normalize_source(args.algorithm_source)
    algos = [a.strip().lower() for a in args.algorithms.split(",") if a.strip()]
    for a in algos:
        if a not in ("ppo", "sac", "td3"):
            print(f"[error] unknown algorithm '{a}'. Supported: ppo, sac, td3", file=sys.stderr)
            sys.exit(2)

    tasks = parse_tasks(args.tasks)
    if not tasks:
        print("[error] --tasks empty", file=sys.stderr)
        sys.exit(2)

    # gpu_sim resolution: explicit CLI > benchmark-spec > false
    if args.gpu_sim == "auto":
        gpu_sim = read_gpu_sim(repo)
    else:
        gpu_sim = args.gpu_sim.lower() in ("true", "1", "yes")
    parallel = gpu_sim
    parallel_suffix = ".parallel" if parallel else ""

    # Algorithm slug → directory name under harbor/scripts/rl/
    if source == "local_implementation":
        algorithm_slug = derive_local_slug(args.algorithm_package)
    else:
        algorithm_slug = source     # 'custom_torch' / 'custom_jax' / 'stable_baseline3'

    project_name = repo.name
    slug = re.sub(r"[^a-z0-9-]+", "-", project_name.lower()).strip("-") or "app"

    common_map: dict[str, str] = {
        "PROJECT_NAME": slug,
        "BENCHMARK_NAME": slug,
        "REPO_PATH": str(repo),
        "DEFAULT_TASK": tasks[0]["id"],
    }
    if source == "local_implementation":
        common_map["LOCAL_IMPL_SLUG"] = algorithm_slug

    # ---- harbor/scripts/rl/<algorithm_slug>/{train,eval,render,visualize,env_wrapper}.py ----
    # `visualize.py` is custom_jax-only (CPU-MuJoCo headed viewer). Other sources skip
    # it silently when the template isn't present.
    src_subtree = TEMPLATES / source / "scripts"
    dst_subtree = repo / "harbor" / "scripts" / "rl" / algorithm_slug
    for fname in ("train.py", "eval.py", "render.py", "env_wrapper.py", "visualize.py"):
        tpl = src_subtree / f"{fname}.template"
        if not tpl.exists():
            continue
        render_template(tpl, dst_subtree / fname, common_map)

    # ---- custom_torch / custom_jax also write algo/, replay/, models/, utils/ subtrees ----
    if source in ("custom_torch", "custom_jax"):
        for sub in ("algo", "replay", "models", "utils"):
            for tpl in (TEMPLATES / source / sub).glob("*.template"):
                out_name = tpl.name.replace(".template", "")
                render_template(tpl, dst_subtree / sub / out_name, common_map)

    # ---- harbor/configs/rl/*.yaml (shared across all sources) ----
    render_template(TEMPLATES / "configs" / "suite.yaml.template",
                    repo / "harbor" / "configs" / "rl" / "suite.yaml", common_map)
    for a in algos:
        render_template(TEMPLATES / "configs" / f"{a}.yaml.template",
                        repo / "harbor" / "configs" / "rl" / f"{a}.yaml", common_map)
        render_template(TEMPLATES / "configs" / f"{a}.parallel.yaml.template",
                        repo / "harbor" / "configs" / "rl" / f"{a}.parallel.yaml", common_map)

    # ---- docker-compose.rl.yaml — best-effort; template was removed when the
    # plugin dropped Docker support, so skip silently if it isn't present. ----
    compose_tpl = TEMPLATES / "docker-compose.rl.yaml.template"
    if compose_tpl.exists():
        compose_text = compose_tpl.read_text()
        for k, v in common_map.items():
            compose_text = compose_text.replace("{{" + k + "}}", v)
        if source == "local_implementation" and args.algorithm_package and Path(args.algorithm_package).exists():
            host_path = str(Path(args.algorithm_package).resolve())
            bind_line = f"      - {host_path}:/workspace/external/{algorithm_slug}"
            compose_text = compose_text.replace("# {{LOCAL_IMPL_MOUNT}}", bind_line)
        else:
            compose_text = compose_text.replace("# {{LOCAL_IMPL_MOUNT}}", "")
        (repo / "docker" / "docker-compose.rl.yaml").parent.mkdir(parents=True, exist_ok=True)
        (repo / "docker" / "docker-compose.rl.yaml").write_text(compose_text)

    # ---- tune.py (top-level cross-impl orchestrator; reads scripts_dir from spec) ----
    tune_src = TEMPLATES / "tune.py.template"
    if tune_src.exists():
        render_template(tune_src, repo / "harbor" / "scripts" / "rl" / "tune.py", common_map)

    # ---- harbor/rl-integration-generator/rl-suite-spec.json ----
    spec_map = {
        "BENCHMARK_NAME": project_name,
        "REPO_PATH": str(repo),
        "TASKS_JSON": json.dumps(tasks)[1:-1],
        "ALGORITHM_SOURCE_KIND": source,
        "ALGORITHM_SLUG": algorithm_slug,
        "ALGORITHM_SOURCE_PACKAGE": args.algorithm_package,
        "PARALLEL": "true" if parallel else "false",
        "PARALLEL_SUFFIX": parallel_suffix,
        "WANDB_ENABLED": "true" if args.wandb_enabled == "true" else "false",
        "WANDB_PROJECT": args.wandb_project,
        "WANDB_ENTITY_JSON": json.dumps(args.wandb_entity) if args.wandb_entity else "null",
        "WANDB_MODE": args.wandb_mode,
    }
    spec_text = (TEMPLATES / "rl-suite-spec.json.template").read_text()
    for k, v in spec_map.items():
        spec_text = spec_text.replace("{{" + k + "}}", v)
    try:
        json.loads(spec_text)
    except json.JSONDecodeError as e:
        print(f"[error] rendered spec is not valid JSON: {e}\n--- text ---\n{spec_text}", file=sys.stderr)
        sys.exit(3)
    (repo / "harbor" / "rl-integration-generator").mkdir(parents=True, exist_ok=True)
    (repo / "harbor" / "rl-integration-generator" / "rl-suite-spec.json").write_text(spec_text)

    # ---- experiments dir scaffolding ----
    (repo / "harbor" / "rl_experiments" / "sweeps").mkdir(parents=True, exist_ok=True)
    (repo / "harbor" / "rl_experiments" / "tunes").mkdir(parents=True, exist_ok=True)

    # ---- <repo>/harbor/rl-integration-generator/rl-integration.md (Layer 6b receipt) ----
    integration_map = dict(common_map)
    integration_map.update({
        "ALGORITHM_SOURCE": source,
        "ALGORITHM_SLUG": algorithm_slug,
        "ALGORITHMS_LIST": ", ".join(algos),
        "DEFAULT_TASK": tasks[0]["id"],
        "GPU_SIM": "true" if gpu_sim else "false",
        "PARALLEL_SUFFIX": parallel_suffix,
        "ENV_FACTORY_MODE": "GPU-batched" if gpu_sim else "CPU vec-env",
        "ENV_FACTORY_DESCRIPTION": (
            "your benchmark's batched factory in env_wrapper.create_env()"
            if gpu_sim else "gymnasium.vector.AsyncVectorEnv + numpy↔torch bridge"
        ),
        "CHECKPOINT_EXT": (
            "zip" if source == "stable_baseline3"
            else "pkl" if source == "custom_jax"
            else "pth"
        ),
        "WANDB_PROJECT": args.wandb_project or "(unset)",
        "WANDB_MODE": args.wandb_mode,
    })
    render_template(TEMPLATES / "rl-integration.md.template",
                    repo / "harbor" / "rl-integration-generator" / "rl-integration.md", integration_map)

    # ---- ensure DataLogger exists at <repo>/harbor/utils/data_logger.py ----
    dl_path = repo / "harbor" / "utils" / "data_logger.py"
    if not dl_path.exists():
        dl_path.parent.mkdir(parents=True, exist_ok=True)
        renderer = PLUGIN_ROOT / "scripts" / "rl-integration-generator" / "render_data_logger.py"
        if renderer.exists():
            subprocess.check_call([
                sys.executable, str(renderer),
                "--backend", "both", "--features", "scalar,image,text",
                "--output", str(dl_path),
            ])
        else:
            print("[warn] harbor/utils/data_logger.py missing AND render_data_logger.py not found.",
                  file=sys.stderr)

    print(json.dumps({
        "ok": True,
        "algorithm_source": source,
        "algorithm_slug": algorithm_slug,
        "gpu_sim": gpu_sim,
        "parallel": parallel,
        "spec": str(repo / "harbor" / "rl-integration-generator" / "rl-suite-spec.json"),
        "scripts_dir": str(repo / "harbor" / "scripts" / "rl" / algorithm_slug),
        "configs": [str(repo / "harbor" / "configs" / "rl" / c) for c in
                    ["suite.yaml"] + [f"{a}.yaml" for a in algos] + [f"{a}.parallel.yaml" for a in algos]],
        "rl_integration_md": str(repo / "harbor" / "rl-integration-generator" / "rl-integration.md"),
    }, indent=2))


if __name__ == "__main__":
    main()
