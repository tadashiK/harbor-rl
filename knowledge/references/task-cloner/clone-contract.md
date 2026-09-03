# task-cloner — Clone Contract

What a clone must guarantee, how each guarantee is checked, and the registration rule. Read by `task-cloner` (and `/harbor:task-clone`).

## The five checks

| # | Check | How it's verified | Hard / soft |
|---|---|---|---|
| SC1 | **Build** — `gym.make(<dest>)` succeeds | `smoke_clone.py` instantiates the env | hard |
| SC2 | **Independence** — every file the surface touches is a DISTINCT copy, and no cloned file still imports the source's `mdp` | static, in `clone_task.py`, and it **fails closed**: no mdp package found, or a surface class defined in no cloned file, is an ERROR — never a skipped check | hard |
| SC3 | **Per-term logging** — `info["detailed_reward"]` present | `smoke_clone.py` reports it after stepping | soft (warn) |
| SC4 | **Rollout** — a few steps run, reward finite every step | `smoke_clone.py` steps 10× and asserts `isfinite` | hard |
| SC5 | **Teardown** — deleting the clone removes its files and the source still builds | `/harbor:task-clone op=delete` removes `cloned_files[]` + the registration anchor, then `gym.make(<source>)` | hard (at delete time) |

SC1 / SC3 / SC4 share one `AppLauncher` in `smoke_clone.py` (sim launch is the expensive part). SC2 is enforced by `clone_task.py` itself. SC5 runs when the clone is deleted, not at create time.

**Why SC2 fails closed.** It used to run only when an mdp package happened to be found. On the
`config/<robot>/` layout most IsaacLab tasks use, it never was — so zero checks ran and the
verdict still read `"independence": "pass"`, while every slot clone shared the family's
`RewardsCfg` and `mdp/rewards.py`. A clone whose independence cannot be verified is a failed
clone, not a passing one.

## Registration rule

- The dest id is `<source>` with a `-rewarditer<NNN>` suffix inserted **before** any `-vN` version token:
  - `Isaac-Lift-Cube-Franka-v0` → `Isaac-Lift-Cube-Franka-rewarditer7-v0`
  - never `Isaac-Lift-Cube-Franka-v0-rewarditer7` (breaks gym version parsing).
- Legal gym id chars only: word chars, `:`, `.`, `-`. **No `#`.**
- Mirror the source's own `gym.register` idiom (same entry-point shape, kwargs), pointing at the cloned cfg class. Do not modify the source's register call.

## Substitution slots (`smoke_clone.py.template`)

| Slot | Value |
|---|---|
| `{{DEST_ID}}` | the cloned gym task id |
| `{{NUM_ENVS}}` | `2` for gpu-sim benchmarks (`benchmark-spec.json:gpu_sim == true`), else `1` |

## What to copy (and what not to)

The surface is **not always beside the registered cfg**. Two layouts ship:

```
flat                              nested (most IsaacLab manipulation tasks)
lift/                             stack_cube/
├── lift_env_cfg.py  <- RewardsCfg  ├── stack_cube_env_cfg.py   <- RewardsCfg
└── mdp/rewards.py                  ├── mdp/rewards.py
                                    └── config/franka/
                                        └── joint_pos_env_cfg.py <- REGISTERED (a subclass)
```

- **Copy:** the `mdp/` package (found by walking UP from the registered cfg to the nearest
  ancestor holding `mdp/__init__.py`), every file at that task root defining a surface class
  (`RewardsCfg`, `ActionsCfg`, `ObservationsCfg`, `EventCfg`, `TerminationsCfg`), and the
  registered cfg itself.
- **Rewire:** relative mdp imports at ANY dot depth (`from . import mdp` and
  `from ... import mdp` both occur), plus the registered cfg's import of the family base so
  the clone subclasses the COPY, not the source.
- **Do NOT copy:** family-wide shared modules the surface never edits — keep importing those
  from their originals.
- `reward_path` in the manifest points at the cloned file that defines `RewardsCfg` — for a
  nested layout that is the family base, not the registered subclass.
- Independence (SC2) is what justifies parallel candidates: two clones editing `rewards.py`
  must edit two different files.
