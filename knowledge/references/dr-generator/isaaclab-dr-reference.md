# IsaacLab domain randomization reference

Loaded by `dr-generator` on demand: the **three randomization groups**, function surface per group,
mode→operation mapping, discovery recipe, defaults, and read-back recipes the smoke uses to verify each
effective term. If a non-IsaacLab family is in play, `task-implementation.md` documents the
family-equivalent surface (e.g. loco-mujoco's `CustomRandomizer`).

---

## The three groups

| # | Group | What it randomizes | Modes | Default |
|---|---|---|---|---|
| 1 | **Robot** | joint stiffness, damping, friction, armature, PD (actuator) gains, base mass, link mass | multiplicative / additive / direct | multiplicative, range `[0.9, 1.1]` |
| 2 | **Object** | object mass, friction, restitution, stiffness, damping, COM — incl. articulated objects + their joints | multiplicative / additive / direct | multiplicative, range `[0.9, 1.1]` |
| 3 | **Observation** | additive noise on every obs term | uniform / gaussian | gaussian, mean 0, std `0.01` |

**Modes (groups 1 & 2) map directly to the IsaacLab `operation` arg:**

| Mode | `operation` | Effect | Sampled value `v` from range |
|---|---|---|---|
| multiplicative | `"scale"` | `prop = default * v` | `v ~ U(lo, hi)` |
| additive | `"add"` | `prop = default + v` | `v ~ U(lo, hi)` |
| direct | `"abs"` | `prop = v` | `v ~ U(lo, hi)` |

The sampling distribution defaults to `"uniform"` (range = `(lo, hi)`). `"log_uniform"` and `"gaussian"`
are also accepted by every physical term — for gaussian the two params are `(mean, std)`, not `(lo, hi)`.
Keep physical DR on **uniform** unless the user asks otherwise.

**Modes (group 3) map to the noise cfg class:**

| Mode | Cfg | Params |
|---|---|---|
| uniform | `UniformNoiseCfg(n_min=-a, n_max=a)` (alias `Unoise`) | symmetric band `±a` |
| gaussian | `GaussianNoiseCfg(mean=0.0, std=s)` (alias `Gnoise`) | `mean=0`, `std=s` |

Import: `from isaaclab.utils.noise import GaussianNoiseCfg as Gnoise, UniformNoiseCfg as Unoise`.

---

## Once per episode per env → `mode="reset"`

DR **must randomize once per episode per env** — a fresh value each time an env resets, held constant
within the episode. In IsaacLab manager-based that is exactly `EventTerm(mode="reset")`.

- `mode="startup"` fires **once for the whole run** (per-env value, but frozen forever) — NOT what we want.
- `mode="reset"` fires **every episode reset, per env** — this is the default for groups 1 & 2.
- `mode="interval"` fires every N seconds — only for ongoing perturbations (kicks/pushes), never the default.

Group 3 (obs noise) is applied per-step by the ObservationManager via the term's `noise=` field; it is
not an EventTerm and has no mode.

> Performance note: `randomize_rigid_body_material` uses CPU tensors and is slow per-reset on huge
> `num_envs`. It is still correct at `reset`. Honor the "once per episode" requirement; mention the cost
> in the receipt if the robot/object has many shapes.

---

## Surface-by-family

| Family | DR surface |
|---|---|
| `isaaclab-manager-based` | `EventCfg` terms (`mode="reset"`) + obs-term `noise=` fields |
| `isaaclab-direct` | hand-coded in `_reset_idx()` (physical) + obs noise in `_get_observations()` |
| `dexteroushands` | `randomization_params` dict in cfg; `apply_randomizations()` inherited |
| `loco-mujoco` | `CustomRandomizer` (`reset()` samples per-episode; see `custom.py`) |
| `dm_control` / `gymnasium-generic` | usually `<unsupported>` — honor the doc |

The rest of this reference is `isaaclab-manager-based`. For other families, mirror the canonical example
that `task-implementation.md` points at.

---

## Function surface (groups 1 & 2)

All live in `isaaclab.envs.mdp` (`from isaaclab.envs import mdp`). Search
`source/isaaclab/isaaclab/envs/mdp/events.py` to confirm symbols in the installed version.

| Function | Group | Targets | Key params (mode arg) | Distribution |
|---|---|---|---|---|
| `randomize_rigid_body_mass` | 1 & 2 | robot bodies / object | `mass_distribution_params=(lo,hi)`, `operation`, `recompute_inertia` | yes |
| `randomize_rigid_body_material` | 1 & 2 | robot / object shapes | `static_friction_range`, `dynamic_friction_range`, `restitution_range`, `num_buckets` | **direct only** (no `operation`) |
| `randomize_actuator_gains` | 1 | robot PD gains | `stiffness_distribution_params`, `damping_distribution_params`, `operation` | yes |
| `randomize_joint_parameters` | 1 & 2 | robot / articulated-object joints | `friction_distribution_params`, `armature_distribution_params`, `lower_limit_distribution_params`, `upper_limit_distribution_params`, `operation` | yes |
| `randomize_fixed_tendon_parameters` | 1 & 2 | tendon-driven articulations | `stiffness_distribution_params`, `damping_distribution_params`, `limit_stiffness_distribution_params`, `operation` | yes |
| `randomize_rigid_body_com` | 1 & 2 | robot bodies / object | `com_range={"x":(lo,hi),"y":...,"z":...}` (additive only) | n/a |
| `randomize_physics_scene_gravity` | scene | gravity vector | `gravity_distribution_params=([lo3],[hi3])`, `operation` | yes |

`randomize_rigid_body_scale` is `mode="usd"` only (before sim plays) and needs `replicate_physics=False`;
treat as out of scope unless the canonical example wires it.

`randomize_rigid_body_material` samples friction/restitution **directly** from the ranges (effectively
"direct" mode); there is no scale/add. To express a multiplicative `[0.9, 1.1]` on friction, set the
range to `(0.9, 1.1)` around the nominal of `1.0`, or to the asset's nominal × `[0.9, 1.1]` if known.
Restitution default `(0.0, 0.0)` (leave unrandomized) unless the task is contact-rich.

### Authoring shape (groups 1 & 2)

```python
# <task>/<task>_env_cfg.py  (or its events.py)
from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

@configclass
class EventCfg:
    # --- §3 reset terms authored by task-generator live here too; DO NOT touch them
    reset_object_position: EventTerm = ...   # left alone
    reset_robot_joints:    EventTerm = ...   # left alone

    # === Group 1 — robot ===
    robot_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.9, 1.1),   # multiplicative
            "operation": "scale",
            "recompute_inertia": True,
        },
    )
    robot_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    robot_joint_params = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.9, 1.1),
            "armature_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    robot_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.9, 1.1),
            "dynamic_friction_range": (0.9, 1.1),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )

    # === Group 2 — object ===
    object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    object_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": (0.9, 1.1),
            "dynamic_friction_range": (0.9, 1.1),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )
    # articulated object joints → same randomize_joint_parameters with asset_cfg("object", joint_names=".*")
```

### Authoring shape (group 3 — obs noise)

Noise goes on the **§5 obs terms**, not in EventCfg:

```python
from isaaclab.utils.noise import GaussianNoiseCfg as Gnoise

@configclass
class PolicyCfg(ObsGroup):
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Gnoise(mean=0.0, std=0.01))
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Gnoise(mean=0.0, std=0.01))
    object_pos = ObsTerm(func=mdp.root_pos_w, noise=Gnoise(mean=0.0, std=0.01),
                         params={"asset_cfg": SceneEntityCfg("object")})
    # ...one noise per active obs term
```

Per-term std may differ by physical scale (positions vs velocities) — start every term at the default
(`std=0.01` gaussian / `±0.01` uniform) unless the user gives per-term values.

---

## Discovery recipe (run before authoring)

Discover the **available** terms by probing the built env, so the receipt can list available vs effective.

```python
base  = env.unwrapped
robot = base.scene["robot"]                     # the articulation
print("robot joints :", robot.joint_names)      # → group-1 joint/actuator/gain targets
print("robot bodies :", robot.body_names)       # → group-1 mass/material targets
print("actuators    :", list(robot.actuators))  # → PD-gain targets (randomize_actuator_gains)

# objects: scan scene for RigidObject / Articulation entries that aren't the robot
from isaaclab.assets import Articulation, RigidObject
for name, ent in base.scene._rigid_objects.items():   # group-2 rigid objects
    print("object(rigid):", name, ent.body_names)
for name, ent in base.scene._articulations.items():   # group-2 articulated objects (+ joints)
    if name == "robot": continue
    print("object(artic):", name, "joints:", ent.joint_names)

# obs terms → group 3
for grp, terms in base.observation_manager.active_terms.items():
    print("obs group", grp, "terms:", terms)
```

A term is **available** if its target exists (joint list non-empty, body has mass, obs term active).
A term is **effective** if you wired it AND the smoke confirms it changes the property (groups 1 & 2)
or the observation (group 3).

---

## Read-back recipes (smoke verification)

The smoke collapses each effective physical term's range to a **point interval `[k, k]`** so the sampled
value is exactly `k` (uniform), then reads the property straight from the physics view and asserts it
equals the original modified by `k`. Run at `num_envs=16`.

```python
robot = base.scene["robot"]; obj = base.scene["object"]

# --- mass (randomize_rigid_body_mass) ---
default = robot.data.default_mass.clone()             # (E, B)
actual  = robot.root_physx_view.get_masses()          # (E, B)
# scale:  actual ≈ default * k    add: default + k    abs: k
torch.testing.assert_close(actual, default * k, rtol=1e-3, atol=1e-4)

# --- material friction/restitution (randomize_rigid_body_material), direct ---
mats = robot.root_physx_view.get_material_properties() # (E, S, 3): [static, dynamic, restitution]
# point interval makes bucketing irrelevant → every shape == k
assert torch.allclose(mats[..., 0], torch.full_like(mats[..., 0], k_static))

# --- actuator gains (randomize_actuator_gains), scale ---
default_k = robot.data.default_joint_stiffness.clone()
actual_k  = robot.data.joint_stiffness.clone()         # refreshed after reset
torch.testing.assert_close(actual_k, default_k * k, rtol=1e-3, atol=1e-4)
# damping: default_joint_damping / joint_damping

# --- joint params (randomize_joint_parameters) ---
# armature: robot.data.joint_armature vs robot.data.default_joint_armature
# friction: robot.data.joint_friction_coeff vs robot.data.default_joint_friction_coeff

# --- COM (randomize_rigid_body_com), additive ---
coms = robot.root_physx_view.get_coms()                # default + offset

# --- object mass/material: same as robot, asset = base.scene["object"] ---
```

Obs-noise terms (group 3) can't be pinned to an exact value. Verify them by building a paired
**no-noise** cfg (set `noise=None` on every obs term) and asserting the noisy rollout diverges from the
clean one, and (for gaussian) that the empirical std over `16 envs × steps` ≈ configured `std` within a
loose factor (e.g. 0.3×–3×).

---

## Defaults when the user description is silent

**Wire every available term (hard constraint).** When the description gives no DR-term restriction, the
default is *comprehensive*: every available term in all three groups becomes effective.

- Groups 1 & 2: **multiplicative, `(0.9, 1.1)`, `mode="reset"`** on EVERY discovered target — mass (link + base), actuator P/D gains, joint friction + armature, material (friction/restitution, direct), COM, and the same for objects + their joints. Not mass-only.
- Group 3: **gaussian `mean=0, std=0.01`** on every active obs term.
- A term is skipped *only* when its target is absent, it would change obs structure, or the range would drive a quantity to ≤0 — and the skip is logged with its reason.

When the description *does* restrict ("only mass", "sensor noise only", "no DR on the robot"), or
`dr_overrides` lists a specific set, honor that narrower set. User-supplied overrides (which term /
mode / range, see agent Inputs) always win over these defaults.

---

## Forbidden

- Ranges that drop a quantity to zero/negative (`mass*0`, friction `< 0`, stiffness `≤ 0` under scale) — env diverges on reset. `randomize_rigid_body_mass`/`actuator_gains` reject `scale` ranges that include 0.
- DR axes that change obs **structure** (enabling/disabling a sensor) — smoke shape check fails.
- `mode="startup"` for groups 1 & 2 when the requirement is once-per-episode (use `reset`).
- Cross-section edits to §1..§6 just to "make DR work" — surface to the user instead.
