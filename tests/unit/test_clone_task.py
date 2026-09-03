"""Unit: scripts/task-cloner/clone_task.py — the isolation primitive behind parallel tuning.

Every parallel reward-tune candidate lives on a clone, so a bug here either orphans files or
— much worse — leaves two candidates sharing an mdp module, which looks like isolation and
silently is not. The properties pinned below are the clone contract's deterministic half
(the sim-side build/rollout checks stay with the agent):

  - the registration rule (suffix BEFORE any -vN, legal gym chars only);
  - SC2 independence — the cloned cfg imports its COPIED mdp, never the source's;
  - the source is never modified except for one appended registration;
  - create rolls back cleanly on failure, and delete reverses a create exactly.
"""
import importlib.util
import json
import subprocess
import sys

import pytest

from _pluginmeta import ROOT

SRC = ROOT / "scripts" / "task-cloner" / "clone_task.py"


def _mod():
    spec = importlib.util.spec_from_file_location("clone_task", SRC)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = _mod()

# A FLAT-layout task: the registered cfg defines its own surface classes beside mdp/.
# (It must actually define RewardsCfg — an earlier version of this fixture only referenced
# it, which let a clone that copied no reward at all look correct.)
CFG_PY = '''\
from . import mdp
from .mdp import rewards

@configclass
class RewardsCfg:
    reach = RewTerm(func=mdp.reach, weight=1.0)

@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    rewards: RewardsCfg = RewardsCfg()
'''

REGISTER_PY = '''\
import gymnasium as gym
from . import lift_env_cfg

gym.register(
    id="Isaac-Lift-Cube-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.lift_env_cfg:LiftEnvCfg"},
    disable_env_checker=True,
)
'''


@pytest.fixture
def repo(tmp_path):
    pkg = tmp_path / "source" / "tasks" / "lift"
    (pkg / "mdp").mkdir(parents=True)
    (pkg / "__init__.py").write_text(REGISTER_PY)
    (pkg / "lift_env_cfg.py").write_text(CFG_PY)
    (pkg / "mdp" / "__init__.py").write_text("from .rewards import *\n")
    (pkg / "mdp" / "rewards.py").write_text("def reach(env):\n    return 0.0\n")
    return tmp_path


# --- the layout that actually ships ------------------------------------------
#
# Real IsaacLab manipulation tasks split across two levels: the family base holds
# RewardsCfg and sits beside mdp/, while the REGISTERED cfg is a thin robot subclass two
# directories down. The flat fixture above is the layout the cloner originally assumed —
# it passed while producing clones that shared the source's reward, because the SC2
# asserts were skipped when no mdp package was found. Found by running against a real
# repo, not by this suite.

BASE_CFG_PY = '''\
from . import mdp

@configclass
class ActionsCfg:
    pass

@configclass
class RewardsCfg:
    reach = RewTerm(func=mdp.reach, weight=1.0)

@configclass
class StackCubeEnvCfg(ManagerBasedRLEnvCfg):
    rewards: RewardsCfg = RewardsCfg()
'''

ROBOT_CFG_PY = '''\
from ... import mdp
from ...stack_cube_env_cfg import StackCubeEnvCfg

@configclass
class FrankaStackCubeEnvCfg(StackCubeEnvCfg):
    pass
'''

ROBOT_REGISTER_PY = '''\
import gymnasium as gym
from . import joint_pos_env_cfg

gym.register(
    id="IsaacLab-Franka-StackCube",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaStackCubeEnvCfg"},
    disable_env_checker=True,
)
'''


@pytest.fixture
def nested_repo(tmp_path):
    """stack_cube/{stack_cube_env_cfg.py, mdp/} + config/franka/joint_pos_env_cfg.py"""
    root = tmp_path / "source" / "tasks" / "stack_cube"
    (root / "mdp").mkdir(parents=True)
    (root / "config" / "franka").mkdir(parents=True)
    (root / "__init__.py").write_text("")
    (root / "stack_cube_env_cfg.py").write_text(BASE_CFG_PY)
    (root / "mdp" / "__init__.py").write_text("from .rewards import *\n")
    (root / "mdp" / "rewards.py").write_text("def reach(env):\n    return 0.0\n")
    (root / "config" / "__init__.py").write_text("")
    (root / "config" / "franka" / "__init__.py").write_text(ROBOT_REGISTER_PY)
    (root / "config" / "franka" / "joint_pos_env_cfg.py").write_text(ROBOT_CFG_PY)
    return tmp_path


def _create_nested(repo, dest="IsaacLab-Franka-StackCube-rslot0"):
    manifest = repo / "manifest.json"
    r = subprocess.run(
        [sys.executable, str(SRC), "--op", "create", "--repo", str(repo),
         "--source", "IsaacLab-Franka-StackCube", "--dest", dest,
         "--surface", "reward,actions", "--manifest", str(manifest)],
        capture_output=True, text=True)
    return r, manifest


def test_nested_layout_copies_the_file_that_defines_the_reward(nested_repo):
    """The registered cfg is a subclass; RewardsCfg lives in the family base. Copying
    only the registered file leaves the clone inheriting the source's reward."""
    r, _ = _create_nested(nested_repo)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    root = nested_repo / "source" / "tasks" / "stack_cube"

    assert (root / "stack_cube_env_cfg_rslot0.py").is_file(), "family base not cloned"
    assert "RewardsCfg" in (root / "stack_cube_env_cfg_rslot0.py").read_text()
    assert out["reward_path"].endswith("stack_cube_env_cfg_rslot0.py"), \
        "reward_path must point at the file defining RewardsCfg, not the robot subclass"


def test_nested_layout_finds_mdp_two_levels_up(nested_repo):
    r, _ = _create_nested(nested_repo)
    assert r.returncode == 0, r.stderr
    root = nested_repo / "source" / "tasks" / "stack_cube"
    assert (root / "mdp_rslot0" / "rewards.py").is_file(), \
        "mdp/ sits at the task root, not beside the registered cfg"


def test_nested_layout_rewires_imports_at_every_dot_depth(nested_repo):
    r, _ = _create_nested(nested_repo)
    assert r.returncode == 0, r.stderr
    root = nested_repo / "source" / "tasks" / "stack_cube"

    base = (root / "stack_cube_env_cfg_rslot0.py").read_text()
    assert "from . import mdp_rslot0 as mdp" in base

    robot = (root / "config" / "franka" / "joint_pos_env_cfg_rslot0.py").read_text()
    assert "from ... import mdp_rslot0 as mdp" in robot, "3-dot mdp import not rewired"
    assert "from ...stack_cube_env_cfg_rslot0 import" in robot, \
        "the clone still subclasses the SOURCE's base cfg"


def test_nested_clone_is_genuinely_independent(nested_repo):
    """The property the whole mechanism exists for: editing the clone cannot reach the
    source. Two parallel candidates must not share a reward file."""
    r, _ = _create_nested(nested_repo)
    assert r.returncode == 0, r.stderr
    root = nested_repo / "source" / "tasks" / "stack_cube"

    (root / "mdp_rslot0" / "rewards.py").write_text("def reach(env):\n    return 999.0\n")
    (root / "stack_cube_env_cfg_rslot0.py").write_text(
        (root / "stack_cube_env_cfg_rslot0.py").read_text().replace("weight=1.0", "weight=42.0"))

    assert "999.0" not in (root / "mdp" / "rewards.py").read_text()
    assert "weight=42.0" not in (root / "stack_cube_env_cfg.py").read_text()


def test_independence_fails_closed_when_no_mdp_exists(nested_repo):
    """The defect that shipped: with no mdp package found, every SC2 assert was skipped
    and the verdict still said pass. An unverifiable clone must be a FAILED clone."""
    root = nested_repo / "source" / "tasks" / "stack_cube"
    __import__("shutil").rmtree(root / "mdp")

    r, manifest = _create_nested(nested_repo)
    assert r.returncode != 0, "a clone that cannot be made independent must not report pass"
    assert "mdp" in (r.stderr + r.stdout).lower()
    assert not manifest.exists(), "wrote a manifest for a clone it could not verify"


def test_nested_delete_restores_the_tree(nested_repo):
    root = nested_repo / "source" / "tasks" / "stack_cube"
    before = {p: p.read_bytes() for p in sorted(root.rglob("*.py"))}

    r, manifest = _create_nested(nested_repo)
    assert r.returncode == 0, r.stderr
    d = subprocess.run(
        [sys.executable, str(SRC), "--op", "delete", "--repo", str(nested_repo),
         "--source", "IsaacLab-Franka-StackCube", "--manifest", str(manifest)],
        capture_output=True, text=True)
    assert d.returncode == 0, d.stderr

    assert {p: p.read_bytes() for p in sorted(root.rglob("*.py"))} == before
    assert not (root / "mdp_rslot0").exists()


def _create(repo, dest="Isaac-Lift-Cube-Franka-rslot0-v0"):
    manifest = repo / "manifest.json"
    r = subprocess.run(
        [sys.executable, str(SRC), "--op", "create", "--repo", str(repo),
         "--source", "Isaac-Lift-Cube-Franka-v0", "--dest", dest,
         "--surface", "reward", "--manifest", str(manifest)],
        capture_output=True, text=True)
    return r, manifest


# --- registration rule -------------------------------------------------------

@pytest.mark.parametrize("source,dest,suffix", [
    ("Isaac-Lift-Cube-Franka-v0", "Isaac-Lift-Cube-Franka-rslot0-v0", "rslot0"),
    ("Isaac-Lift", "Isaac-Lift-abtest1", "abtest1"),
])
def test_suffix_goes_before_the_version_token(source, dest, suffix):
    assert M.derive_suffix(source, dest) == suffix


@pytest.mark.parametrize("dest,why", [
    ("Isaac-Lift-Cube-Franka-v0-rslot0", "suffix after -vN breaks gym version parsing"),
    ("Isaac-Lift-Cube-Franka-rslot#0-v0", "'#' is not a legal gym id char"),
    ("Isaac-Lift-Cube-Franka-v0", "dest must differ from source"),
    ("Totally-Different-v0", "dest must be source + one suffix"),
])
def test_illegal_dest_ids_are_refused(dest, why):
    with pytest.raises(SystemExit):
        M.derive_suffix("Isaac-Lift-Cube-Franka-v0", dest)


# --- create ------------------------------------------------------------------

def test_create_produces_an_independent_clone(repo):
    r, manifest_path = _create(repo)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["status"] == "pass"
    assert out["checks"] == {"id_rule": "pass", "independence": "pass", "registered": "pass"}

    pkg = repo / "source" / "tasks" / "lift"
    cloned_cfg = pkg / "lift_env_cfg_rslot0.py"
    assert cloned_cfg.is_file()

    # SC2: the clone reads its OWN mdp copy. This is what makes two candidates safe to
    # edit concurrently — if it still imported `.mdp`, they would share reward code.
    text = cloned_cfg.read_text()
    assert "mdp_rslot0" in text
    assert "from .mdp import" not in text
    assert "from . import mdp\n" not in text
    assert (pkg / "mdp_rslot0" / "rewards.py").is_file()

    # The copies are real files, not links back to the source.
    (pkg / "mdp_rslot0" / "rewards.py").write_text("def reach(env):\n    return 1.0\n")
    assert "0.0" in (pkg / "mdp" / "rewards.py").read_text(), "editing the clone hit the source"


def test_create_only_appends_a_registration_to_the_source(repo):
    before = (repo / "source" / "tasks" / "lift" / "lift_env_cfg.py").read_text()
    r, _ = _create(repo)
    assert r.returncode == 0, r.stderr
    pkg = repo / "source" / "tasks" / "lift"
    assert (pkg / "lift_env_cfg.py").read_text() == before, "source cfg was modified"
    reg = (pkg / "__init__.py").read_text()
    assert reg.startswith(REGISTER_PY), "the source's own registration was rewritten"
    assert "Isaac-Lift-Cube-Franka-rslot0-v0" in reg


def test_manifest_lists_every_created_file(repo):
    r, manifest_path = _create(repo)
    assert r.returncode == 0, r.stderr
    manifest = json.loads(manifest_path.read_text())
    assert manifest["dest_id"] == "Isaac-Lift-Cube-Franka-rslot0-v0"
    for rel in manifest["cloned_files"]:
        assert (repo / rel).exists(), f"manifest lists a file that was not created: {rel}"
    assert any("mdp_rslot0" in f for f in manifest["cloned_files"])
    assert manifest["reward_path"].endswith("lift_env_cfg_rslot0.py")


def test_create_refuses_an_unknown_source(repo):
    r = subprocess.run(
        [sys.executable, str(SRC), "--op", "create", "--repo", str(repo),
         "--source", "Isaac-Not-Registered-v0", "--dest", "Isaac-Not-Registered-x-v0",
         "--manifest", str(repo / "m.json")],
        capture_output=True, text=True)
    assert r.returncode != 0
    assert not (repo / "m.json").exists(), "wrote a manifest for a clone it never made"


# --- delete ------------------------------------------------------------------

def test_delete_reverses_create_exactly(repo):
    pkg = repo / "source" / "tasks" / "lift"
    before = {p: p.read_bytes() for p in sorted(pkg.rglob("*.py"))}

    r, manifest_path = _create(repo)
    assert r.returncode == 0, r.stderr

    d = subprocess.run(
        [sys.executable, str(SRC), "--op", "delete", "--repo", str(repo),
         "--source", "Isaac-Lift-Cube-Franka-v0", "--manifest", str(manifest_path)],
        capture_output=True, text=True)
    assert d.returncode == 0, d.stderr

    after = {p: p.read_bytes() for p in sorted(pkg.rglob("*.py"))}
    assert after == before, "delete did not restore the source tree byte-for-byte"
    assert not (pkg / "mdp_rslot0").exists()
