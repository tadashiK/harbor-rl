# IsaacLab-Franka-HangMug — Implementation Spec

- robot: Franka Emika Panda (7 DoF arm + parallel gripper)
- simulator: IsaacLab (Isaac Sim, manager-based)
- objects: mug, single-peg mug rack, lab table
- bimanual: false
- summary: Pick up a mug and hang it by its handle on a peg rack.

Build form — `gym.make('<task>')` with no `cfg` raises `TypeError: ManagerBasedRLEnv.__init__() missing 1 required positional argument: 'cfg'` for every IsaacLab manager-based task, so build through `parse_env_cfg`:

```python
from isaaclab_tasks.utils import parse_env_cfg

cfg = parse_env_cfg("IsaacLab-Franka-HangMug", device="cuda:0", num_envs=2)
env = gym.make("IsaacLab-Franka-HangMug", cfg=cfg)
```

> **NOTE — the repo carries a modified `RewardManager`.** `source/isaaclab/isaaclab/managers/reward_manager.py` has harbor's dt-strip applied: `value = func(...) * weight` (no `* dt`) and `_step_reward[:, i] = value` (no `/ dt`). **All §6 weights below are nominal PER-STEP magnitudes, not per-second.** Reproducing §6 on a stock IsaacLab (`* dt`, dt = 0.05) divides every term by 20 and destroys the ladder's calibration against the sparse landmarks. Apply the dt-strip or multiply every weight by 20.

---

## §1 Registration + Scene

### Description

A Franka Panda stands at the env origin facing +X on a lab table. A mug stands upright on the table 42 cm in front of it with its handle pointing world +Y; a single-peg mug rack stands 20 cm further out, kinematic, with its peg pointing world −X (back toward the robot) and angled 25.6° upward. The policy must grasp the mug body, lift it, thread the peg through the handle hole, release, and retract. Two fingertip contact sensors (filtered against the mug) and two frame transformers (fingertip TCP, peg hang point) complete the scene. There is no `CommandsCfg` — the goal is implicit in the rack's pose.

### Decisions resolved

| Knob | Value |
|---|---|
| Task IDs | `IsaacLab-Franka-HangMug`, `IsaacLab-Franka-HangMug-Play` |
| Entry point | `isaaclab.envs:ManagerBasedRLEnv`, `disable_env_checker=True` |
| Package path | `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/hang_mug/` |
| Robot | `FRANKA_PANDA_HIGH_PD_CFG` (stock IsaacLab Panda) with `spawn.activate_contact_sensors = True` |
| Robot init | `pos=(0, 0, 0)`, joints `(0, −0.300, 0, −2.400, 0, 2.100, −0.8298)`, fingers `0.04` |
| TCP offset | `panda_hand` + `(0, 0, 0.1034)` — same value for `ee_frame` and the IK `body_offset` |
| Table | `harbor/assets/table/lab_table_instanceable_colored_rotated.usd` at `(0.32, 0, 0)`; 0.844 × 1.25 m, visual surface at z = 0 |
| **Work-surface datum** | `TABLE_SURFACE_Z = -0.0077` (see the asset-provenance block below) |
| Ground plane | `/World/GroundPlane` at z = −0.82 |
| Light | `DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0)` |
| Mug | `harbor/assets/mug_rack/mug.usd`, `pos=(0.42, −0.04, −0.0077)`, `rot=(0.7071068, 0, 0, 0.7071068)` (yaw +90°), mass 0.06 kg, `solver_position_iteration_count=16`, contact sensors on |
| Rack | `harbor/assets/mug_rack/rack.usd`, `pos=(0.62, 0, 0)`, identity rot, **kinematic** (`kinematic_enabled=True`, `disable_gravity=True`), contact sensors off |
| Frame transformers | `ee_frame` (TCP) and `peg_frame` (hang point `RACK_HANG_POINT_LOCAL`), both sourced at `panda_link0` |
| Contact sensors | `finger_left_contact` / `finger_right_contact` on the Panda fingertips, `filter_prim_paths_expr=["{ENV_REGEX_NS}/Mug"]`, `history_length=1` |
| Timing | `decimation=6`, `sim.dt=1/120`, `episode_length_s=10.0` → 200 control steps @ 20 Hz |
| Scene | `num_envs=4096`, `env_spacing=2.5`, `replicate_physics=False` |
| Commands | `None` |

### Asset provenance (LOAD-BEARING — read before regenerating anything)

**Both meshes are BAKED to scale; they are NOT spawned scaled.** The cfg spawns both USDs with the default `scale=(1, 1, 1)`, and the mesh vertices already carry the scale factor. A reproducer that applies the scale again at spawn time gets a mug at 0.64 and a task whose every hardcoded constant is wrong.

Source meshes: <https://github.com/ankile/robust-rearrangement/tree/main/src/assets/furniture/mesh/mug_rack>
- `muggood.obj` → `mug.obj` → `mug.usd`, **baked scale 0.80**
- `mugrack/mugrack.obj` → `rack.obj` → `rack.usd`, **baked scale 1.00**

Exact regeneration command (relative to the repo root):

```bash
.venv/bin/python harbor/assets/mug_rack/prepare_meshes.py \
    --src <dir holding muggood.obj + mugrack.obj> \
    --mug-scale 0.80 \
    --rack-scale 1.0
```

> **NOTE — `--rack-scale 1.0` is MANDATORY and OVERRIDES the script's own default.**
> `harbor/assets/mug_rack/prepare_meshes.py` declares `MUG_SCALE = 0.80` and `RACK_SCALE = 1.15`. The shipped rack was built with an explicit `--rack-scale 1.0`. Regenerating without the flag yields a rack 15 % larger and silently invalidates every constant in `mdp/geometry.py` (peg root/tip, hang point, post top) — the task still builds and still trains, it just never succeeds. `--mug-scale 0.80` matches the script default but should be passed explicitly for the same reason.

**Why mug scale 0.80 specifically.** Four inequalities have to hold at once and they close a narrow window (recorded in `harbor/create-task/isaaclab-franka-hangmug/task-history.md`, §1 "Decisions resolved", lines ~184–208):

1. the mug's rim must pass between the open Franka fingers → rim diameter `2 × 0.0350 = 6.99 cm < 8.0 cm` opening → **mug ≤ 0.80**; mug 0.85 fails this;
2. the peg must fit the handle hole with real slack → hole 2.0 cm − peg 1.11 cm = ±5.2 mm;
3. the peg's horizontal run must exceed (finger reach ≈ 4.5 cm + threading 1.8 cm) → 7.2 cm ✓;
4. the hand must ride above the post at the hang pose → hand z 0.386 vs post top 0.334 ✓; rack 1.15 leaves only 7 mm here.

**`harbor/assets/mug_rack/geometry.json`, verbatim** — every constant in `mdp/geometry.py` is a measurement OF this bake and is meaningless against a different one:

```json
{
  "mug": {
    "scale": 0.8,
    "height": 0.104611792,
    "rim_radius": 0.03495219535559948,
    "bbox_min": [
      -0.03599,
      -0.03812,
      0.0
    ],
    "bbox_max": [
      0.05746,
      0.03624,
      0.10461
    ],
    "handle_azimuth_before_deg": 139.08921821907472
  },
  "rack": {
    "scale": 1.0,
    "peg_root": [
      -0.03089,
      0.00097,
      0.2163
    ],
    "peg_tip": [
      -0.0889,
      0.00097,
      0.2441
    ],
    "peg_radius": 0.00481,
    "peg_length": 0.06432,
    "peg_elevation_deg": 25.611,
    "post_top_z": 0.33453952,
    "post_x_extent": [
      -0.039865656446801526,
      0.012598610539325051
    ],
    "base_plate_radius": 0.08287371043148177,
    "base_plate_top_z": 0.034972340000000005,
    "peg_azimuth_before_deg": -19.024939793293488
  }
}
```

**Collision decomposition (CoACD, pre-decomposed — NOT handed to the simulator's cooker).** Both USDs carry one `convexHull` collider prim per CoACD part. Identical params for both assets:

| asset | source | source sha256 | `max_convex_hull` | `threshold` | `preprocess_resolution` | hulls | collider prims on stage |
|---|---|---|---|---|---|---|---|
| mug | `mug.obj` | `90ece9324384bc3b825a13e77ccb330212f5b5748f2177072ab12553533b86c1` | 32 | 0.05 | 50 | **32** | 32 |
| rack | `rack.obj` | `50202a47ddafe4ed05f700e665e3eb1b4ecfef5bdd3c2bbbb99f4d00d8124da0` | 32 | 0.05 | 50 | **3** | 3 |

The mug's 32 hulls are what keeps the handle hole OPEN in physics: the union of the hulls leaves a 2.0 × 2.4 cm opening. Handing the mug mesh to PhysX as one prim produces a convex blob with no hole and the task becomes unsolvable while every smoke still passes. The rack's 3 parts are base plate / post / peg.

Manifests on disk: `harbor/assets/mug_rack/{mug,rack}.coacd.json` (params + part list) and `{mug,rack}.build.json` (`ok`, `parts`, `collider_prims`, `approximations: ["convexHull"]`).

**`TABLE_SURFACE_Z = -0.0077` — the measured work-surface datum, and it is NOT the table's visual top.** The shared lab table asset spawns with a `convexHull` collider whose cooked surface sits **7.7 mm below** the visual tabletop. Measured by resting the mug at eight (x, y) points across the workspace: every one settles at root z = −0.0077 ± 0.0001, and the value does not move with the mug's mass (0.06 vs 0.5 kg), the solver iteration count (16 vs 64), the physics dt (1/120 vs 1/480), or explicit contact/rest offsets — so it is geometry, not solver residual. The same mug resting on the analytic ground plane penetrates only 0.24 mm, which confirms the offset belongs to the table asset and not to the mug's CoACD hulls. **Everything that rests on the table is referenced to this datum, not to z = 0** — including `MUG_INIT_POS` and `lift_distance`'s `init_z`.

**PhysX broadphase capacity.** The shipped value is:

```python
self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024
```

> **NOTE — the shipped value is undersized and should be `>= 256 * 1024`.**
> This scene has 32 (mug) + 3 (rack) collider prims per env; at 4096 envs PhysX requests up to 143360 aggregate pairs and logs capacity warnings. A read-only A/B on the same trained checkpoint (recorded in `harbor/create-task/isaaclab-franka-hangmug/spec.json:phases.reward_tune.post_hoc_physics_check`) measured:
>
> | arm | mean_return | std_return | mean_length | n | capacity warnings |
> |---|---|---|---|---|---|
> | A — `64*1024` (shipped) | 1416.661 | 1112.718 | 127.734 | 5698 | 520 |
> | B — `256*1024` | 2292.206 | 10.099 | 80.181 | 4002 | 0 |
>
> Conclusion: the undersized broadphase **suppressed measured success**; the recorded `success_rate = 0.522` is a pessimistic artifact. The fix was NOT applied because it would change the task cfg the verdict was scored under. A reproducer must know both the shipped number and that it is wrong.

### Code

`source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/hang_mug/config/franka/__init__.py` (registration):

```python
import gymnasium as gym

gym.register(
    id="IsaacLab-Franka-HangMug",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaHangMugEnvCfg",
    },
)

gym.register(
    id="IsaacLab-Franka-HangMug-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaHangMugEnvCfg_PLAY",
    },
)
```

`hang_mug/__init__.py`:

```python
"""Single-arm mug-hanging environments (Franka Panda + Franka hand)."""

from . import mdp  # noqa: F401 — re-exports task-local helpers
```

`hang_mug/config/__init__.py`:

```python
"""Per-robot configurations for the hang_mug environment."""
```

`hang_mug/mdp/__init__.py`:

```python
"""MDP terms for the hang_mug task."""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .actions import *  # noqa: F401, F403
from .actions_cfg import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403
```

`hang_mug/hang_mug_env_cfg.py` — scene + env cfg (MDP groups appear in their own sections below):

```python
"""Abstract (robot-agnostic) cfg groups for the single-arm `hang_mug` task.

A mug stands upright on the table; a single-peg mug rack stands in front of it, its peg
pointing back toward the robot and angled 25.6 deg upward. The policy must grasp the mug by
its body, lift it, thread the peg through the handle hole, let go, and retract — the episode
ends the moment the mug hangs on the peg with the gripper clear of it.

The concrete robot articulation, the mug, the rack, the frame transformers and the action stack
are filled by the per-robot subclass in `config/franka/joint_pos_env_cfg.py`.
"""

from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp

##
# Scene definition
##

_TABLE_USD_PATH = str(
    Path(__file__).resolve().parents[6]
    / "harbor" / "assets" / "table" / "lab_table_instanceable_colored_rotated.usd"
)

# Table centre. The lab table is 0.844 x 1.25 m with its surface at z = 0, so this places the
# top face over x in [-0.10, 0.74] — the robot base at x = 0 sits on it, and so does the rack
# at x = 0.62 (base plate reaches 0.70).
TABLE_POS = (0.32, 0.0, 0.0)


@configclass
class HangMugSceneCfg(InteractiveSceneCfg):
    # Filled by the per-robot subclass via __post_init__.
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    mug: RigidObjectCfg = MISSING
    rack: RigidObjectCfg = MISSING
    peg_frame: FrameTransformerCfg = MISSING

    # Fingertip contact sensors, filtered against the mug. §4's `gripper_clear_of_mug` reads
    # them to decide whether the gripper has actually let go; §6 gets a grasp gate for free.
    finger_left_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
        update_period=0.0, history_length=1, debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Mug"],
    )
    finger_right_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
        update_period=0.0, history_length=1, debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Mug"],
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=list(TABLE_POS)),
        spawn=UsdFileCfg(usd_path=_TABLE_USD_PATH),
    )
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -0.82]),
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# Environment configuration
##


@configclass
class HangMugEnvCfg(ManagerBasedRLEnvCfg):
    """Abstract single-arm hang_mug env cfg (robot-agnostic)."""

    scene: HangMugSceneCfg = HangMugSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # No CommandsCfg — the goal is implicit in the scene (the rack peg).
    commands: object | None = None

    def __post_init__(self):
        self.decimation = 6
        self.episode_length_s = 10.0                 # 200 control steps at 20 Hz
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
```

`hang_mug/config/franka/joint_pos_env_cfg.py` — per-robot scene + action wiring (verbatim, whole file):

```python
"""Franka Panda configuration for the single-arm `hang_mug` task.

Fills the abstract `HangMugEnvCfg` with the stock IsaacLab Franka Panda articulation, the mug,
the single-peg mug rack, the EE / peg frame transformers, and the EMA EE-delta + binary gripper
action stack.

Layout (world = env-local; the robot root sits at the env origin, facing +X):

    robot   (0.00,  0.00, 0.00)   Franka Panda, tool pointing straight down, fingers closing
                                  along world X
    table   (0.32,  0.00, 0.00)   surface at z = 0
    mug     (0.42, -0.04, 0.00)   upright, handle pointing world +Y
    rack    (0.62,  0.00, 0.00)   peg pointing world -X (back toward the robot), 25.6 deg up

Why the mug's handle points +Y: the peg is threaded through the handle hole along the hole's
axis, which is the mug's local Y. With the handle at world +Y the hole's axis is world X — the
same axis the peg runs along, and the same axis the fingers close on, so the fingers grip the
mug at +/-X where the handle is not.
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions import DifferentialInverseKinematicsActionCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip

from ... import mdp
from ...hang_mug_env_cfg import HangMugEnvCfg
from ...mdp.geometry import RACK_HANG_POINT_LOCAL

_HARBOR_ASSETS = Path(__file__).resolve().parents[8] / "harbor" / "assets"
_MUG_USD_PATH = str(_HARBOR_ASSETS / "mug_rack" / "mug.usd")
_RACK_USD_PATH = str(_HARBOR_ASSETS / "mug_rack" / "rack.usd")

# Franka TCP: `panda_hand` origin to the fingertip pads. Same value for the ee_frame sensor and
# the IK action's body_offset — they must name the same point or the IK tracks a frame the
# reward cannot see.
FRANKA_TCP_OFFSET = (0.0, 0.0, 0.1034)

# Home pose. Ported from the insert-drawer base's shape (joints 1/3/5 zeroed so the arm stays
# in the robot's x-z plane) and then re-solved for THIS layout, because the action term locks
# the EE orientation at whatever the reset pose produces:
#   * joint6 = 2.100 puts the tool axis within 1 deg of straight down (the stock Franka home
#     pose leaves it tilted 45 deg forward, which no top-down grasp survives);
#   * joint7 = -0.8298 rotates the finger-closing axis onto world +X, so the fingers grip the
#     mug body on the two sides the handle is not on.
# Measured at this pose: TCP = (0.447, 0.000, 0.329), tool tilt 0.9 deg, finger axis
# (1.000, 0.019, -0.016).
FRANKA_INIT_JOINT_POS = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.300,
    "panda_joint3": 0.0,
    "panda_joint4": -2.400,
    "panda_joint5": 0.0,
    "panda_joint6": 2.100,
    "panda_joint7": -0.8298,
    "panda_finger_joint.*": 0.04,
}

# MEASURED work-surface height, and it is NOT the table's visual top.
#
# The inherited lab table spawns with a `convexHull` collider whose cooked surface sits 7.7 mm
# below the visual tabletop. Measured by resting the mug at eight (x, y) points across the
# workspace: every one settles at root z = -0.0077 +/- 0.0001, and the value does not move with
# the mug's mass (0.06 vs 0.5 kg), the solver iteration count (16 vs 64), the physics dt
# (1/120 vs 1/480), or explicit contact/rest offsets — so it is geometry, not solver residual.
# The same mug resting on the ANALYTIC ground plane penetrates 0.24 mm, which is what confirms
# the offset belongs to the table asset and not to the mug's CoACD hulls: mug-vs-peg contact,
# the one this task's tolerances actually depend on, is accurate to a fraction of a millimetre.
#
# Everything that rests on the table is therefore referenced to this datum rather than to z = 0.
TABLE_SURFACE_Z = -0.0077

MUG_INIT_POS = (0.42, -0.04, TABLE_SURFACE_Z)
MUG_INIT_ROT = (0.7071068, 0.0, 0.0, 0.7071068)   # yaw +90 deg: mug local +X (handle) -> world +Y
MUG_MASS = 0.06

RACK_INIT_POS = (0.62, 0.0, 0.0)
RACK_INIT_ROT = (1.0, 0.0, 0.0, 0.0)

# The stock cfg does not enable contact reporting; the fingertip sensors in §1 need it.
FRANKA_HANGMUG_CFG = FRANKA_PANDA_HIGH_PD_CFG.copy()
FRANKA_HANGMUG_CFG.spawn.activate_contact_sensors = True


@configclass
class FrankaHangMugEnvCfg(HangMugEnvCfg):
    """Franka Panda mug-hanging environment."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = FRANKA_HANGMUG_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos=FRANKA_INIT_JOINT_POS, pos=(0.0, 0.0, 0.0)
            ),
        )

        # Mug — CoACD-decomposed USD (32 convexHull parts, handle hole preserved; see
        # harbor/assets/mug_rack/mug.coacd.json). The mesh is pre-scaled to 0.80, so spawn
        # scale stays (1, 1, 1).
        self.scene.mug = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Mug",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=list(MUG_INIT_POS), rot=list(MUG_INIT_ROT)
            ),
            spawn=UsdFileCfg(
                usd_path=_MUG_USD_PATH,
                activate_contact_sensors=True,
                mass_props=sim_utils.MassPropertiesCfg(mass=MUG_MASS),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Rack — kinematic, so the policy cannot knock the goal over and the peg pose is a
        # constant of the episode. 3 CoACD parts: base plate, post, peg.
        self.scene.rack = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Rack",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=list(RACK_INIT_POS), rot=list(RACK_INIT_ROT)
            ),
            spawn=UsdFileCfg(
                usd_path=_RACK_USD_PATH,
                activate_contact_sensors=False,
                rigid_props=RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                    max_depenetration_velocity=1000.0,
                ),
            ),
        )

        # EE frame (fingertip TCP) — same offset as the IK action's body_offset.
        ee_marker_cfg = FRAME_MARKER_CFG.copy()
        ee_marker_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
        ee_marker_cfg.prim_path = "/Visuals/EEFrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=True,
            visualizer_cfg=ee_marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=list(FRANKA_TCP_OFFSET)),
                )
            ],
        )

        # Peg frame — the target hang point on the peg, drawn so the reset / keyframe renders
        # show where the handle hole has to end up.
        peg_marker_cfg = FRAME_MARKER_CFG.copy()
        peg_marker_cfg.markers["frame"].scale = (0.06, 0.06, 0.06)
        peg_marker_cfg.prim_path = "/Visuals/PegFrameTransformer"
        self.scene.peg_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=True,
            visualizer_cfg=peg_marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Rack",
                    name="peg_hang_point",
                    offset=OffsetCfg(pos=list(RACK_HANG_POINT_LOCAL)),
                )
            ],
        )

        # Action stack — 3-D EMA xyz EE-delta (orientation locked) + binary gripper.
        #
        # The workspace box is a hard guarantee, not a hint: x <= 0.56 keeps the hand's far
        # edge (TCP + ~4.5 cm along the finger axis) clear of the post's near face at
        # x = 0.608, so no policy can drive the gripper into the rack. It still leaves 2.9 cm
        # of travel past the peg tip (x = 0.531) where 1.8 cm is what a hang needs.
        self.actions.arm_action = mdp.EMACumulativeDeltaPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=FRANKA_TCP_OFFSET),
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls"
            ),
            scale=(0.01, 0.01, 0.01),
            alpha=0.5,
            pos_lower_limit=[0.30, -0.20, 0.02],
            pos_upper_limit=[0.56, 0.20, 0.40],
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )


@configclass
class FrankaHangMugEnvCfg_PLAY(FrankaHangMugEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
```

`hang_mug/mdp/geometry.py` — the baked-geometry constants and the frame helpers §4/§5/§6 all read (verbatim, whole file):

```python
"""Baked geometry of the mug / mug-rack assets, and the frame helpers that read it.

Both meshes were re-framed and scaled once, ahead of time, by
`harbor/assets/mug_rack/prepare_meshes.py`, so the shipped USDs spawn at scale (1, 1, 1) and
every constant below is a direct measurement of the shipped mesh (see
`harbor/assets/mug_rack/geometry.json`).

Mug frame  (`harbor/assets/mug_rack/mug.usd`, upstream `muggood.obj` at scale 0.80)
    origin on the mug axis at the base plane, +Z up, handle pointing +X.

Rack frame (`harbor/assets/mug_rack/rack.usd`, upstream `mugrack.obj` at scale 1.00)
    origin on the post axis at the base plane, +Z up, peg pointing -X and 25.6 deg upward.

The task hangs the mug by threading the peg through the handle hole, so the two frames meet at
exactly two points: the hole centre in the mug frame and the hang point on the peg in the rack
frame. Everything §4 and §5 need is derived from those two.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ----------------------------------------------------------------------------- mug (scale 0.80)
MUG_HEIGHT = 0.1046
"""Mug height (m): base plane to rim."""

MUG_RIM_RADIUS = 0.0350
"""Outer rim radius (m). 2 x 0.0350 = 7.0 cm < the Franka hand's 8 cm opening, which is what
makes a top-down grasp of the body possible at all."""

MUG_HANDLE_HOLE_LOCAL = (0.0400, 0.0, 0.0680)
"""Centre of the handle hole in the mug frame (m). The hole's axis is the mug's local Y."""

MUG_HOLE_HALF_WIDTH = 0.0052
"""Clearance (m) around the hole centre along the mug's local X, measured on the COLLISION
meshes (CoACD parts), minus the peg radius. This is the lateral precision the policy needs."""

MUG_HOLE_HALF_HEIGHT = 0.0119
"""Same, along the mug's local Z."""

MUG_GRASP_LOCAL_Z = 0.080
"""Nominal grasp height on the mug body (m above the base plane); body diameter there is
6.0 cm, so the fingers travel ~1 cm per side from the open position."""

# ---------------------------------------------------------------------------- rack (scale 1.00)
RACK_PEG_ROOT_LOCAL = (-0.0309, 0.0010, 0.2163)
"""Where the peg leaves the post, in the rack frame (m)."""

RACK_PEG_TIP_LOCAL = (-0.0889, 0.0010, 0.2441)
"""Free end of the peg, in the rack frame (m). The mug is threaded on from here."""

RACK_PEG_RADIUS = 0.0048
"""Peg radius (m)."""

RACK_HANG_POINT_LOCAL = (-0.0727, 0.0010, 0.2519)
"""Target point ON the peg, 1.8 cm from the tip along the peg axis (m). The handle bar is
1.24 cm thick, so a hole centred here has the peg fully through it with ~3 mm to spare on each
side, while leaving 5.9 cm between the mug axis and the post for the gripper."""

RACK_POST_TOP_Z = 0.3345
"""Top of the vertical post (m). The hand rides above this during the hang."""


def _asset_frame_point(env: "ManagerBasedRLEnv", asset_name: str, local: tuple) -> torch.Tensor:
    """A point given in an asset's local frame, expressed in world coordinates."""
    asset: RigidObject = env.scene[asset_name]
    offset = torch.tensor(local, dtype=torch.float32, device=env.device).expand(env.num_envs, 3)
    return asset.data.root_pos_w[:, :3] + quat_apply(asset.data.root_quat_w, offset)


def mug_hole_pos_w(env: "ManagerBasedRLEnv", mug_name: str = "mug") -> torch.Tensor:
    """World position of the handle-hole centre, (num_envs, 3)."""
    return _asset_frame_point(env, mug_name, MUG_HANDLE_HOLE_LOCAL)


def peg_hang_point_w(env: "ManagerBasedRLEnv", rack_name: str = "rack") -> torch.Tensor:
    """World position of the target hang point on the peg, (num_envs, 3)."""
    return _asset_frame_point(env, rack_name, RACK_HANG_POINT_LOCAL)


def peg_segment_w(env: "ManagerBasedRLEnv", rack_name: str = "rack"):
    """World endpoints of the peg axis: (root, tip), each (num_envs, 3)."""
    return (
        _asset_frame_point(env, rack_name, RACK_PEG_ROOT_LOCAL),
        _asset_frame_point(env, rack_name, RACK_PEG_TIP_LOCAL),
    )


def point_to_peg_axis(env: "ManagerBasedRLEnv", point_w: torch.Tensor, rack_name: str = "rack"):
    """Perpendicular distance from `point_w` to the peg axis, and where along the peg it lands.

    Returns ``(distance, t)`` with ``t = 0`` at the peg root and ``t = 1`` at the tip, both
    (num_envs,). ``t`` is NOT clamped: a point beyond the tip reports ``t > 1``, which is how
    §4 distinguishes "threaded onto the peg" from "floating past its end".
    """
    root, tip = peg_segment_w(env, rack_name)
    axis = tip - root
    length_sq = (axis * axis).sum(dim=-1).clamp(min=1e-9)
    t = ((point_w - root) * axis).sum(dim=-1) / length_sq
    closest = root + t.unsqueeze(-1) * axis
    return torch.norm(point_w - closest, dim=-1), t
```

### Smoke — §1

```bash
cd <repo>
.venv/bin/python harbor/create-task/isaaclab-franka-hangmug/smokes/smoke_s1.py
```

Expected verdict (`smokes/smoke_s1.verdict.json`, `status: pass`):

```
C1 registration + build      PASS  built IsaacLab-Franka-HangMug with num_envs=2 (requested 2)
C2 action space              PASS  single_action_space (4,) vs terms {'arm_action': 3, 'gripper_action': 1} (total_action_dim=4)
C3 observation space         PASS  policy: declared (32,) actual (32,)
C4 max episode length        PASS  max_episode_length=200, expected ceil(10.0/0.05)=200
C5 timing                    PASS  sim.dt=0.008333333333333333 decimation=6 step_dt=0.05 (20.0 Hz policy, 120.0 Hz physics), episode_length_s=10.0 => 200.0000 steps
C6 collision geometry        PASS  3 assets inspected
C7 articulation limits       PASS  no limited joints outside ['robot'] — nothing articulated to test
C8 collision provenance      PASS  robot: primitive/exact by decision — skipped; mug: coacd manifest 32 hulls, 32 collider prim(s) on stage; rack: coacd manifest 3 hulls, 3 collider prim(s) on stage
```

Direct build check — write this to a file and run it with `<repo>/.venv/bin/python` (the Isaac Sim app must be launched before `isaaclab_tasks` is imported, so it cannot be a one-liner):

```python
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args([])
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym          # noqa: E402
import isaaclab_tasks            # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg   # noqa: E402

TASK = "IsaacLab-Franka-HangMug"
cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=2)
env = gym.make(TASK, cfg=cfg)
print(f"obs={env.observation_space}  act={env.action_space}")
env.close()
app.close()
```

Canonical stdout, measured at `num_envs=2`:

```
obs=Dict('policy': Box(-inf, inf, (2, 32), float32))  act=Box(-inf, inf, (2, 4), float32)
```

---

## §2 Actions

### Description

4-D action: a 3-D EMA-smoothed **cumulative** EE position delta with the EE **orientation locked** at the post-reset value, plus a 1-D binary gripper. The arm term is a custom subclass of IsaacLab's `DifferentialInverseKinematicsAction`: the policy emits `(dx, dy, dz)` in `[-1, 1]`, which is scaled by `0.01` per axis, accumulated across the episode, clamped **at the source** against a hard workspace box, EMA-blended (`alpha = 0.5`) with the previously applied target, re-clamped, and forwarded to the DLS IK controller as an absolute 7-D pose target. There is no rotation channel at all — the reset home pose (tool 0.9° off vertical, fingers closing along world +X) is the orientation for the whole episode, which is why the home pose in §1 had to be re-solved rather than inherited.

### Decisions resolved

| Knob | Value |
|---|---|
| `arm_action` class | `mdp.EMACumulativeDeltaPositionActionCfg` → `EMACumulativeDeltaPositionAction` |
| `action_dim` (arm) | 3 (`dx, dy, dz`); orientation locked |
| `asset_name` / `joint_names` / `body_name` | `"robot"` / `["panda_joint.*"]` / `"panda_hand"` |
| `body_offset` | `DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.1034))` |
| `controller` | `DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")` |
| `scale` | `(0.01, 0.01, 0.01)` — **note the cfg class default is `(0.02, 0.02, 0.02)`; this task overrides it** |
| `alpha` | `0.5` |
| `pos_lower_limit` | `[0.30, -0.20, 0.02]` |
| `pos_upper_limit` | `[0.56, 0.20, 0.40]` — `x ≤ 0.56` keeps the hand's far edge clear of the post's near face at `x = 0.608`, while leaving 2.9 cm of travel past the peg tip (`x = 0.531`) where 1.8 cm is what a hang needs |
| `forbidden_xy_half` | `None` (unused by this task) |
| `gripper_action` class | `mdp.BinaryJointPositionActionCfg` (stock IsaacLab) |
| gripper joints / commands | `["panda_finger.*"]`, open `0.04`, close `0.0` |
| total action dim | **4** |

### Code

`hang_mug/hang_mug_env_cfg.py` — `ActionsCfg`:

```python
@configclass
class ActionsCfg:
    """3-D xyz EMA-smoothed EE-delta (orientation locked) + binary gripper. Fields are filled
    by the franka subclass `__post_init__`."""

    arm_action: "mdp.EMACumulativeDeltaPositionActionCfg" = MISSING
    gripper_action: "mdp.BinaryJointPositionActionCfg" = MISSING
```

(the concrete wiring is in the `joint_pos_env_cfg.py` block in §1, under "Action stack")

`hang_mug/mdp/actions_cfg.py` (verbatim, whole file):

```python
"""Cfg for `EMACumulativeDeltaPositionAction` — fixed-orientation (position-only) variant.

Inherits `DifferentialInverseKinematicsActionCfg`, so `asset_name`,
`joint_names`, `body_name`, `body_offset`, `controller` all behave the same.
The `controller` field MUST have `command_type="pose"` and
`use_relative_mode=False`.

Extras beyond the parent:
    scale               3-element per-axis position scale (sx, sy, sz).
    alpha               EMA weight in [0, 1]. 1.0 = no smoothing.
    pos_lower_limit     Optional per-axis clamp on the post-EMA position
    pos_upper_limit     target. Both must be set together, else no clamp.
    forbidden_xy_half   Optional no-go square half-extent (None for this task).
"""
from __future__ import annotations

from isaaclab.envs.mdp.actions import DifferentialInverseKinematicsActionCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from .actions import EMACumulativeDeltaPositionAction


@configclass
class EMACumulativeDeltaPositionActionCfg(DifferentialInverseKinematicsActionCfg):
    class_type: type[ActionTerm] = EMACumulativeDeltaPositionAction
    scale: tuple[float, float, float] = (0.02, 0.02, 0.02)
    alpha: float = 0.5
    pos_lower_limit: list[float] | None = None
    pos_upper_limit: list[float] | None = None
    forbidden_xy_half: float | None = None
```

`hang_mug/mdp/actions.py` (verbatim, whole file):

```python
"""EMA cumulative-delta task-space (EE position) action — fixed-orientation variant.

The policy emits ONLY a 3-D position delta `(dx, dy, dz)`. The EE quaternion is
FIXED at the post-reset value (no rotation channel, no axis-angle compose).

Behavior:
    1. Accumulates the 3-D position delta over the episode:
           delta_t = clamp(delta_{t-1} + s * a_t,
                           pos_lower - init_ee_pos, pos_upper - init_ee_pos)
       The clamp is applied at the SOURCE so a policy reversal takes effect on
       the very next step (no accumulated overshoot to burn through).
    2. Anchors on the EE pose captured lazily at the first `process_actions`
       after `env.reset()`:
           abs_pos_t  = init_ee_pos + delta_t
           abs_quat_t = init_ee_quat                       (orientation locked)
    3. EMA-smooths against the previously applied target on the position
       channel only, then re-clamps defensively.
    4. Optionally pushes the target out of a `|x| <= h and |y| <= h` no-go
       square (`forbidden_xy_half`); unused by this task (`None`).
    5. Forwards the absolute 7-D pose target to the IK controller (forced into
       `use_relative_mode=False`) and reuses the parent's `apply_actions()`.
"""
from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg


class EMACumulativeDeltaPositionAction(DifferentialInverseKinematicsAction):
    cfg: "actions_cfg.EMACumulativeDeltaPositionActionCfg"

    def __init__(self, cfg: "actions_cfg.EMACumulativeDeltaPositionActionCfg", env: "ManagerBasedEnv"):
        if cfg.controller.use_relative_mode:
            raise ValueError(
                "EMACumulativeDeltaPositionAction handles relative deltas itself; "
                "set controller.use_relative_mode=False"
            )
        if cfg.controller.command_type != "pose":
            raise ValueError(f"requires command_type='pose'; got '{cfg.controller.command_type}'")
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(env.num_envs, 3, device=env.device)
        self._processed_actions = torch.zeros(env.num_envs, 7, device=env.device)
        self._scale = torch.zeros((env.num_envs, 3), device=env.device)
        self._scale[:] = torch.tensor(cfg.scale, device=env.device)
        if not 0.0 <= cfg.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1]. Got {cfg.alpha}.")
        self._alpha = cfg.alpha
        self.del_action = torch.zeros((env.num_envs, 3), device=env.device)
        self.init_ee_pos = torch.zeros((env.num_envs, 3), device=env.device)
        self.init_ee_quat = torch.zeros((env.num_envs, 4), device=env.device)
        self.init_ee_quat[:, 0] = 1.0
        self._prev_applied_pos = torch.zeros((env.num_envs, 3), device=env.device)
        self._needs_reanchor = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.pos_lower_limit = (
            torch.tensor(cfg.pos_lower_limit, device=self.device) if cfg.pos_lower_limit is not None else None
        )
        self.pos_upper_limit = (
            torch.tensor(cfg.pos_upper_limit, device=self.device) if cfg.pos_upper_limit is not None else None
        )
        self.forbidden_xy_half: float | None = getattr(cfg, "forbidden_xy_half", None)

    @property
    def action_dim(self) -> int:
        return 3

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._needs_reanchor[:] = True
            self.del_action[:] = 0.0
        else:
            self._needs_reanchor[env_ids] = True
            self.del_action[env_ids] = 0.0

    def process_actions(self, actions: torch.Tensor) -> None:
        if self._needs_reanchor.any():
            ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
            mask = self._needs_reanchor
            self.init_ee_pos[mask] = ee_pos_curr[mask]
            self.init_ee_quat[mask] = ee_quat_curr[mask]
            self._prev_applied_pos[mask] = ee_pos_curr[mask]
            self._needs_reanchor[:] = False
        actions = torch.clamp(actions, -1.0, 1.0)
        self._raw_actions[:] = actions
        scaled = actions * self._scale
        self.del_action += scaled
        if self.pos_lower_limit is not None and self.pos_upper_limit is not None:
            del_lower = self.pos_lower_limit - self.init_ee_pos
            del_upper = self.pos_upper_limit - self.init_ee_pos
            self.del_action = torch.clamp(self.del_action, del_lower, del_upper)
        abs_pos = self.init_ee_pos + self.del_action
        ema_pos = self._alpha * abs_pos + (1.0 - self._alpha) * self._prev_applied_pos
        if self.pos_lower_limit is not None and self.pos_upper_limit is not None:
            ema_pos = torch.clamp(ema_pos, self.pos_lower_limit, self.pos_upper_limit)
        if self.forbidden_xy_half is not None:
            h = self.forbidden_xy_half
            inside = (ema_pos[:, 0].abs() <= h) & (ema_pos[:, 1].abs() <= h)
            if inside.any():
                # Push the target out of the |x|<=h and |y|<=h square to the nearest edge.
                dx = h - ema_pos[:, 0].abs()
                dy = h - ema_pos[:, 1].abs()
                push_x = dx <= dy
                sign_x = torch.sign(ema_pos[:, 0])
                sign_x = torch.where(sign_x == 0, torch.ones_like(sign_x), sign_x)
                sign_y = torch.sign(ema_pos[:, 1])
                sign_y = torch.where(sign_y == 0, torch.ones_like(sign_y), sign_y)
                new_x = torch.where(inside & push_x, sign_x * h, ema_pos[:, 0])
                new_y = torch.where(inside & (~push_x), sign_y * h, ema_pos[:, 1])
                ema_pos = torch.stack([new_x, new_y, ema_pos[:, 2]], dim=-1)
        self._processed_actions[:, :3] = ema_pos
        self._processed_actions[:, 3:7] = self.init_ee_quat
        self._prev_applied_pos[:] = ema_pos
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        self._ik_controller.set_command(self._processed_actions, ee_pos_curr, ee_quat_curr)
```

### Smoke — §2 and §2.5

```bash
cd <repo>
.venv/bin/python harbor/create-task/isaaclab-franka-hangmug/smokes/smoke_s2.py
.venv/bin/python harbor/create-task/isaaclab-franka-hangmug/smokes/smoke_s2_5.py
```

Expected stdout (verdict summaries, both `status: pass`):

```
S2   OK: mode=ema_delta_ee_pose action 0.2 → EE pos delta ≈ 0.0017 as expected
S2.5 OK: held 150 steps, actuator tracks target (max err 0.0000 rad, mean 0.0000 rad < tol 0.05) across 2 envs
```

`smoke_s2.py` is run with `ACTION_MODE = "ema_delta_ee_pose"` and `ACTION_TERM_NAME = "arm_action"`; `smoke_s2_5.py` with `ROBOT_ASSET = "robot"`, `SETTLE_STEPS = 150`, `TRACKING_TOL = 0.05`, `NUM_ENVS = 2`.

---

## §3 Reset

### Description

Reset is near-deterministic. The robot's joints are reset to their configured home pose exactly (`position_range=(1.0, 1.0)` is a multiplicative scale of the default joint positions, i.e. no randomization, zero velocity). The mug is re-placed at its configured pose with a small ±2 cm uniform jitter in x and y and no z jitter — the z offset is the measured table datum baked into `MUG_INIT_POS`, so jittering it would sink or float the mug. The rack is re-placed at exactly its configured pose (all ranges zero); the term exists so the kinematic rack is explicitly re-zeroed each episode rather than inheriting whatever the previous episode left. No orientation jitter on either object: the mug's yaw is what puts the handle hole's axis onto the peg's axis, so randomizing it would make many episodes geometrically unsolvable under the orientation-locked action stack.

### Decisions resolved

| Term | mode | func | params |
|---|---|---|---|
| `reset_robot_joints` | `reset` | `mdp.reset_joints_by_scale` | `position_range=(1.0, 1.0)`, `velocity_range=(0.0, 0.0)` |
| `reset_mug` | `reset` | `mdp.reset_root_state_uniform` | `pose_range={"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (0.0, 0.0)}`, `velocity_range={}`, `asset_cfg=SceneEntityCfg("mug")` |
| `reset_rack` | `reset` | `mdp.reset_root_state_uniform` | `pose_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}`, `velocity_range={}`, `asset_cfg=SceneEntityCfg("rack")` |

Effective mug spawn box: `x ∈ [0.40, 0.44]`, `y ∈ [-0.06, -0.02]`, `z = -0.0077`, yaw fixed at +90°.

### Code

`hang_mug/hang_mug_env_cfg.py` — `EventCfg` (verbatim):

```python
@configclass
class EventCfg:
    """Reset terms only (§3). §7 DR was not requested for this task."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )
    reset_mug = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("mug"),
        },
    )
    reset_rack = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("rack"),
        },
    )
```

All three funcs are stock IsaacLab (`isaaclab.envs.mdp`), re-exported through `hang_mug/mdp/__init__.py`.

### Smoke — §3

```bash
cd <repo>
.venv/bin/python harbor/create-task/isaaclab-franka-hangmug/smokes/smoke_s3.py          # numeric
.venv/bin/python harbor/create-task/isaaclab-franka-hangmug/smokes/smoke_s3_render.py   # visual
```

Expected verdicts (both `status: pass`):

```
S3        C1 injected values      PASS  17/17 injected values match
          C2 buffers populated    PASS  1 articulation(s), 2 rigid object(s), 4 sensor(s)
          C3 robot configuration  PASS  robot: 9 joints, range [-2.400, +2.100]
          C4 layout settles       PASS  2 object(s) held within 0.02 m over 30 steps
          C5 repeatable           PASS  two seed=0 resets agree across 3 entities
S3-visual V1 frames captured      PASS  4/4 resets produced a frame
          V2 frames are not blank PASS  pixel std per frame: [51.5, 46.4, 46.3, 46.3] (>5 means something is in view)
```

Reset frames land in `harbor/create-task/isaaclab-franka-hangmug/smoke_s3_frames/` (`reset_00..03.png`). C4 is the check that catches the datum: with `z = 0` instead of `-0.0077` the mug is ejected on the first physics step.

---

## §4 Goal + Termination

### Description

There is no `CommandsCfg` — the goal is the rack peg's pose, which is a constant of the episode because the rack is kinematic. Success is the user's criterion verbatim ("the mug is hanged and the gripper is not touched anymore"), split into two predicates that only ever succeed together: `mug_on_peg` (the hole centre is within `radial_tol` of the peg **axis** with its projection between root `t = 0` and tip `t = 1`, the mug's base plane is ≥ 10 cm above the table, and both linear and angular velocity are near zero) AND `gripper_clear_of_mug` (neither fingertip reports contact force above threshold **and** the TCP has retreated ≥ 15 cm from the mug root). The contact test alone is insufficient — a gripper parked open around the mug reports no force while plainly not having let go — so the distance term is what makes "released" mean released. There are **no failure terminations**: a dropped mug costs the policy the rest of the episode's progress rather than the episode itself, keeping the failure signal on the reward side where §6 can shape it.

`hung_but_gripper_holding` is not a termination term; it is the negative predicate S4 uses to prove the release half of `hang_success` is load-bearing.

### Decisions resolved

| Term | func | params | `time_out` |
|---|---|---|---|
| `time_out` | `mdp.time_out` | — | `True` |
| `success` | `mdp.hang_success` | `radial_tol=0.020`, `min_height=0.10`, `min_ee_distance=0.15`, `contact_force_threshold=0.01` | `False` |

`commands = None`. `mug_on_peg` internal defaults: `radial_tol=0.020`, `t_min=-0.05`, `t_max=1.0`, `min_height=0.10`, `max_lin_vel=0.05`, `max_ang_vel=1.0`.

Measured reference poses at a true hang (env-local, from `smokes/smoke_s4.py`): mug root `(0.5473, -0.0390, 0.1839)`, hole centre `(0.5473, 0.0010, 0.2519)` == `peg_hang_point`, TCP holding the mug there `(0.5499, -0.0340, 0.2634)`.

### Code

`hang_mug/hang_mug_env_cfg.py` — `TerminationsCfg` (verbatim):

```python
@configclass
class TerminationsCfg:
    """`time_out` plus the success term. No failure terminations: as in the insert-drawer base,
    a dropped mug costs the policy the rest of the episode's progress rather than the episode
    itself, so the failure signal stays on the reward side where §6 can shape it."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=mdp.hang_success,
        time_out=False,
        params={
            "radial_tol": 0.020,
            "min_height": 0.10,
            "min_ee_distance": 0.15,
            "contact_force_threshold": 0.01,
        },
    )
```

`hang_mug/mdp/terminations.py` (verbatim, whole file):

```python
"""Goal predicates and terminations for the hang_mug task (§4).

The task is done when the mug HANGS on the rack peg and the gripper has let go — the user's
success criterion verbatim ("the mug is hanged and the gripper is not touched anymore"). That
splits into two predicates that are checked separately and only ever succeed together:

    mug_on_peg           the peg passes through the handle hole, the mug is held up in the air
                         by it, and it has stopped moving
    gripper_clear_of_mug neither fingertip reports contact with the mug AND the fingertip TCP
                         has retreated from it

`gripper_clear_of_mug` is trivially TRUE at reset (the arm starts nowhere near the mug), so it
is not a standalone subgoal — it is the second conjunct of `hang_success`, and S4 tests it
through the negative predicate `hung_but_gripper_holding`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, FrameTransformer

from .geometry import mug_hole_pos_w, point_to_peg_axis

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def mug_on_peg(
    env: "ManagerBasedRLEnv",
    radial_tol: float = 0.020,
    t_min: float = -0.05,
    t_max: float = 1.0,
    min_height: float = 0.10,
    max_lin_vel: float = 0.05,
    max_ang_vel: float = 1.0,
    mug_cfg: SceneEntityCfg = SceneEntityCfg("mug"),
) -> torch.Tensor:
    """True where the mug hangs on the peg, in the geometric sense.

    Three conditions, each closing a hole the other two leave open:

    * ``radial``  — the handle-hole centre sits within ``radial_tol`` of the peg AXIS, and its
      projection lands between the peg's root (``t = 0``) and its tip (``t = 1``). The hole is
      2.0 x 2.4 cm on the collision meshes, so once the peg is through it, the centre can never
      be further than ~1.5 cm from the axis; a mug standing NEXT to the rack is metres away in
      ``t`` or ``radial``, whichever it misses first.
    * ``supported`` — the mug's base plane is ``min_height`` above the table. Nothing in this
      scene can hold a mug 10 cm up except the peg: the rack's own base plate tops out at
      3.5 cm, and the gripper is excluded by the second half of ``hang_success``.
    * ``settled`` — linear and angular velocity are near zero. Without it a mug in mid-flight
      that happens to pass the peg would qualify for the one frame it is there.
    """
    mug: RigidObject = env.scene[mug_cfg.name]
    dist, t = point_to_peg_axis(env, mug_hole_pos_w(env, mug_cfg.name))
    radial = (dist < radial_tol) & (t > t_min) & (t < t_max)

    base_z = mug.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    supported = base_z > min_height

    settled = (torch.norm(mug.data.root_lin_vel_w, dim=-1) < max_lin_vel) & (
        torch.norm(mug.data.root_ang_vel_w, dim=-1) < max_ang_vel
    )
    return radial & supported & settled


def gripper_clear_of_mug(
    env: "ManagerBasedRLEnv",
    min_ee_distance: float = 0.15,
    contact_force_threshold: float = 0.01,
    mug_cfg: SceneEntityCfg = SceneEntityCfg("mug"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """True where the gripper is neither touching the mug nor still around it.

    Contact alone is not enough: a gripper parked open around the mug reports no contact force
    while plainly not having let go. The distance term is what makes "released" mean released.
    """
    mug: RigidObject = env.scene[mug_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    far = torch.norm(ee_pos_w - mug.data.root_pos_w[:, :3], dim=-1) > min_ee_distance

    left: ContactSensor = env.scene["finger_left_contact"]
    right: ContactSensor = env.scene["finger_right_contact"]
    left_f = torch.norm(left.data.force_matrix_w[:, 0, 0, :], dim=-1)
    right_f = torch.norm(right.data.force_matrix_w[:, 0, 0, :], dim=-1)
    no_contact = (left_f < contact_force_threshold) & (right_f < contact_force_threshold)
    return far & no_contact


def hung_but_gripper_holding(
    env: "ManagerBasedRLEnv",
    radial_tol: float = 0.020,
    min_height: float = 0.10,
    min_ee_distance: float = 0.15,
) -> torch.Tensor:
    """Mug threaded on the peg while the gripper is still on it — success must NOT fire here.

    Not a termination term. It exists so S4 can prove the release half of `hang_success` is
    load-bearing rather than decorative.
    """
    return mug_on_peg(env, radial_tol=radial_tol, min_height=min_height) & ~gripper_clear_of_mug(
        env, min_ee_distance=min_ee_distance
    )


def hang_success(
    env: "ManagerBasedRLEnv",
    radial_tol: float = 0.020,
    min_height: float = 0.10,
    min_ee_distance: float = 0.15,
    contact_force_threshold: float = 0.01,
) -> torch.Tensor:
    """`DoneTerm` func: the mug hangs on the peg AND the gripper has let go of it."""
    return mug_on_peg(env, radial_tol=radial_tol, min_height=min_height) & gripper_clear_of_mug(
        env, min_ee_distance=min_ee_distance, contact_force_threshold=contact_force_threshold
    )
```

### Smoke — §4

```bash
cd <repo>
.venv/bin/python harbor/create-task/isaaclab-franka-hangmug/smokes/smoke_s4.py
```

Expected verdict (`status: pass`, 9 checks):

```
G1 goal representation                                              PASS  4/4 goal checks pass
C1 predicate fires: mug threaded on peg                             PASS  start=clear, forced=2/2
V1 keyframe: mug threaded on peg                                    PASS  01_mug_threaded_on_peg.png (pixel std 46.3)
C2 predicate fires: hung but gripper still on it (success withheld) PASS  start=clear, forced=2/2
V2 keyframe: hung but gripper still on it (success withheld)        PASS  02_hung_but_gripper_still_on_it__success_withheld.png (pixel std 46.3)
C3 predicate fires: success: hung and gripper clear                 PASS  start=clear, forced=2/2
V3 keyframe: success: hung and gripper clear                        PASS  03_success__hung_and_gripper_clear.png (pixel std 46.6)
C4 predicate fires: time-out                                        PASS  start=clear, forced=2/2
V4 keyframe: time-out                                               PASS  04_time_out.png (pixel std 46.8)
```

Keyframes land in `harbor/create-task/isaaclab-franka-hangmug/smoke_s4_frames/`. The C2/V2 pair is the load-bearing one: it proves success is withheld while the gripper is still on a correctly hung mug.

---

## §5 Observation

### Description

A single `policy` group, 32-D, concatenated, every spatial quantity expressed in the **robot root frame**. Term order is the observation layout and must not be reordered. `mug_hole_position` and `peg_hang_point` are derived quantities rather than raw poses: the whole task is "bring those two points together along the peg axis", and a policy that has to re-derive them from a mug quaternion plus a hardcoded rack offset is solving a harder problem than the one being asked — both remain functions of state the robot could measure. `mug_velocity` exists because §4's success predicate has a settled-velocity conjunct; without it the task would be partially observable by accident. `enable_corruption = True` is set but every `Unoise` is `(0.0, 0.0)`, so the observation is currently noise-free — the corruption switch is wired and ready for §7 to populate.

### Decisions resolved

| # | Term | func | dim | slice | noise | params |
|---|---|---|---|---|---|---|
| 1 | `ee_pose` | `mdp.ee_pose_in_robot_root_frame` | 7 | `0:7` | `Unoise(0.0, 0.0)` | — |
| 2 | `mug_pose` | `mdp.mug_pose_in_robot_root_frame` | 7 | `7:14` | `Unoise(0.0, 0.0)` | — |
| 3 | `mug_velocity` | `mdp.mug_velocity_in_robot_root_frame` | 6 | `14:20` | `Unoise(0.0, 0.0)` | — |
| 4 | `mug_hole_position` | `mdp.mug_hole_position_in_robot_root_frame` | 3 | `20:23` | `Unoise(0.0, 0.0)` | — |
| 5 | `peg_hang_point` | `mdp.peg_hang_point_in_robot_root_frame` | 3 | `23:26` | `Unoise(0.0, 0.0)` | — |
| 6 | `gripper_joint_pos` | `mdp.joint_pos` | 2 | `26:28` | `Unoise(0.0, 0.0)` | `asset_cfg=SceneEntityCfg("robot", joint_names=["panda_finger.*"])` |
| 7 | `last_action` | `mdp.last_action` | 4 | `28:32` | — | — |

`__post_init__`: `enable_corruption = True`, `concatenate_terms = True`. **Total obs dim = 32.** No masking / zero-gating anywhere in `mdp/observations.py`. The `_PLAY` variant sets `enable_corruption = False`.

> **NOTE — stale docstring.** The module docstring of `mdp/observations.py` says "26-D policy observation" while the term list right below it sums to 32 (7+7+6+3+3+2+4). The **code is correct at 32**; the leading number is a leftover from an earlier draft. Do not "fix" the layout to match the number.

### Code

`hang_mug/hang_mug_env_cfg.py` — `ObservationsCfg` (verbatim):

```python
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """32-D, robot-root framed. Order is the observation layout — do not reorder."""

        ee_pose = ObsTerm(func=mdp.ee_pose_in_robot_root_frame, noise=Unoise(n_min=0.0, n_max=0.0))
        mug_pose = ObsTerm(func=mdp.mug_pose_in_robot_root_frame, noise=Unoise(n_min=0.0, n_max=0.0))
        mug_velocity = ObsTerm(
            func=mdp.mug_velocity_in_robot_root_frame, noise=Unoise(n_min=0.0, n_max=0.0)
        )
        mug_hole_position = ObsTerm(
            func=mdp.mug_hole_position_in_robot_root_frame, noise=Unoise(n_min=0.0, n_max=0.0)
        )
        peg_hang_point = ObsTerm(
            func=mdp.peg_hang_point_in_robot_root_frame, noise=Unoise(n_min=0.0, n_max=0.0)
        )
        gripper_joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_finger.*"])},
            noise=Unoise(n_min=0.0, n_max=0.0),
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
```

`hang_mug/mdp/observations.py` (verbatim, whole file):

```python
"""Observation helpers for the hang_mug task (§5).

26-D policy observation, every spatial term in the ROBOT ROOT frame:

    ee_pose            (7)  fingertip TCP position + orientation
    mug_pose           (7)  mug base-centre position + orientation
    mug_velocity       (6)  mug linear + angular velocity
    mug_hole_position  (3)  centre of the handle hole — the point that must reach the peg
    peg_hang_point     (3)  target point on the rack peg
    gripper_joint_pos  (2)  finger openings
    last_action        (4)

`mug_hole_position` and `peg_hang_point` are derived quantities rather than raw poses: the whole
task is "bring those two points together along the peg axis", and a policy that has to
re-derive them from a mug quaternion and a hardcoded rack offset is solving a harder problem
than the one being asked. Both are still functions of state the robot could measure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import quat_apply, quat_conjugate, subtract_frame_transforms

from .geometry import mug_hole_pos_w, peg_hang_point_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _to_robot_root(env: "ManagerBasedRLEnv", pos_w: torch.Tensor) -> torch.Tensor:
    robot: RigidObject = env.scene["robot"]
    pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, pos_w)
    return pos_b


def ee_pose_in_robot_root_frame(
    env: "ManagerBasedRLEnv",
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """7-D end-effector pose [x, y, z, qw, qx, qy, qz] in the robot root frame."""
    robot: RigidObject = env.scene["robot"]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        ee_frame.data.target_pos_w[..., 0, :],
        ee_frame.data.target_quat_w[..., 0, :],
    )
    return torch.cat([ee_pos_b, ee_quat_b], dim=-1)


def mug_pose_in_robot_root_frame(
    env: "ManagerBasedRLEnv",
    mug_cfg: SceneEntityCfg = SceneEntityCfg("mug"),
) -> torch.Tensor:
    """7-D mug pose in the robot root frame. The quaternion matters here in a way it does not
    for a cube task: the handle's azimuth decides whether the hole can ever meet the peg."""
    robot: RigidObject = env.scene["robot"]
    mug: RigidObject = env.scene[mug_cfg.name]
    pos_b, quat_b = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        mug.data.root_pos_w[:, :3],
        mug.data.root_quat_w,
    )
    return torch.cat([pos_b, quat_b], dim=-1)


def mug_velocity_in_robot_root_frame(
    env: "ManagerBasedRLEnv",
    mug_cfg: SceneEntityCfg = SceneEntityCfg("mug"),
) -> torch.Tensor:
    """6-D mug root velocity (linear, angular) rotated into the robot root frame.

    Present because §4's success predicate has a settled-velocity conjunct: without this term
    the policy could not tell a hung mug from one that is merely passing the peg, and the task
    would be partially observable by accident.
    """
    robot: RigidObject = env.scene["robot"]
    mug: RigidObject = env.scene[mug_cfg.name]
    root_quat_inv = quat_conjugate(robot.data.root_quat_w)
    lin_b = quat_apply(root_quat_inv, mug.data.root_lin_vel_w)
    ang_b = quat_apply(root_quat_inv, mug.data.root_ang_vel_w)
    return torch.cat([lin_b, ang_b], dim=-1)


def mug_hole_position_in_robot_root_frame(
    env: "ManagerBasedRLEnv",
    mug_cfg: SceneEntityCfg = SceneEntityCfg("mug"),
) -> torch.Tensor:
    """Centre of the mug's handle hole, in the robot root frame."""
    return _to_robot_root(env, mug_hole_pos_w(env, mug_cfg.name))


def peg_hang_point_in_robot_root_frame(
    env: "ManagerBasedRLEnv",
    rack_cfg: SceneEntityCfg = SceneEntityCfg("rack"),
) -> torch.Tensor:
    """Target point on the rack peg, in the robot root frame."""
    return _to_robot_root(env, peg_hang_point_w(env, rack_cfg.name))
```

### Smoke — §5

```bash
cd <repo>
.venv/bin/python harbor/create-task/isaaclab-franka-hangmug/smokes/smoke_s5.py
```

Expected stdout (verdict `status: pass`):

```
S5 OK: 7 terms, order + shapes match, value check verified
```

`EXPECTED_TERMS_PY` for the reproduction:

```python
[("ee_pose", (7,)), ("mug_pose", (7,)), ("mug_velocity", (6,)),
 ("mug_hole_position", (3,)), ("peg_hang_point", (3,)),
 ("gripper_joint_pos", (2,)), ("last_action", (4,))]
```

Expected term/shape record is also on disk at `harbor/create-task/isaaclab-franka-hangmug/smokes/expected_obs.json`.

---

## §6 Reward

### Description

Composer = **sum**, 9 terms, all positive (no penalties, no regularizers). The ladder is the insert-drawer base's shape ported 1:1: a dense pick-and-place stream (`reach_mug` → `is_lifted` → `lift_distance` → `align_stage` → `thread`) that is **zero-masked** once a one-shot geometric latch fires, then a post-latch dense stage (`hold_on_peg` + `release_retract`) gated on the LIVE `mug_on_peg` predicate, then the terminal `success_bonus` paired with §4's success `DoneTerm`.

Two task-specific deviations from the base: the base's single place-attractor is split into a coarse `align_stage` (to a staging point 3.5 cm beyond the hang point along the peg's outward direction — a peg is entered end-on, not dropped into from above) plus a fine `thread` (to the hang point itself); and the base's `close_drawer` stage is replaced by the `hold_on_peg` + `release_retract` pair.

**Declaration order matters.** The pre-latch terms read the latch bit BEFORE `on_peg_latch` writes it, so the frame the latch fires still pays the dense ladder. Reordering `RewardsCfg` changes the reward.

**Latch semantics.** `on_peg_latch` is the only writer. It fires `+1.0` on the FIRST frame the mug is geometrically on the peg (settled, held up, hole on the peg axis, tightened `radial_tol=0.015`), then holds the per-`(env, key)` bit until the next episode reset, detected via `env.episode_length_buf <= 1`. The buffer is a module-level dict keyed by `(id(env), "on_peg_once")`.

**Heights are env-local**: `env.scene.env_origins[:, 2]` is subtracted from any world z before it is compared against a threshold, exactly as `terminations.mug_on_peg` does.

### Decisions resolved

| # | Term | func | weight | gate |
|---|---|---|---|---|
| 1 | `reach_mug` | `mdp.reach_mug` | 0.02 | `(1 − latch)` |
| 2 | `is_lifted` | `mdp.is_lifted` | 0.2 | `(1 − latch)` |
| 3 | `lift_distance` | `mdp.lift_distance` | 0.3 | both fingertip contacts > 1e-3 N AND `(1 − latch)` |
| 4 | `align_stage` | `mdp.align_stage` | 2.0 | mug base z > 0.10 |
| 5 | `thread` | `mdp.thread` | 6.0 | mug base z > 0.10 |
| 6 | `on_peg_latch` | `mdp.on_peg_latch` | 200.0 | one-shot per episode |
| 7 | `hold_on_peg` | `mdp.hold_on_peg` | 4.0 | — (live `mug_on_peg`) |
| 8 | `release_retract` | `mdp.release_retract` | 6.0 | live `mug_on_peg` |
| 9 | `success_bonus` | `mdp.success_bonus` | 2000.0 | — |

All terms use their function defaults (no `params` overrides in `RewardsCfg`). Success term for scoring: `success_bonus`. Composer: `sum`.

### Planning budget (verbatim from `iter_000/design.json:reward.budget_rationale`)

> Per-step income at saturation, in nominal weight units (composer = sum, sign positive = good). Stage ladder: reach 0.02 < is_lifted 0.2 < lift_distance 0.3 < align_stage 2.0 < thread 6.0, so the pre-hang dense ceiling is 8.52/step (episodic ceiling ~1700 over 200 steps, only reachable by a policy that is already threading). Post-hang: hold_on_peg 4.0 + release_retract 6.0 + the residual align_stage/thread payment at the settled hang (~1.1) = ~11.1/step, and at the instant the latch fires (hole on the hang point, mug still gripped) income is 6.0 + 1.3 + 4.0 = 11.3/step — i.e. per-step income never decreases across the stage boundary (reward-experience #8). Sparse landmarks sit above the dense stream: on_peg_latch +200 one-shot (~24x the pre-hang per-step rate) and success_bonus +2000 one-shot, collected on the terminal frame because the §4 success DoneTerm has time_out=False. Farming check (reward-experience #12/#19): the best non-terminating strategy is to thread and hold, worth 200 + 11.3 * (200 - t_thread) <= ~1780 for any physically reachable t_thread >= 60, against ~2250 for threading, releasing and retracting — so finishing strictly dominates camping. Regularizers: none, as in the base.

### Measured outcome (iter 0, promoted as-is)

- `best_success_rate = 0.522`, `best_total_return = 1213.36`, `peak_success_rate = 0.834` at step 31,916,032, 1 tuning iteration.
- See the §1 PhysX note: the 0.522 figure is a **pessimistic artifact** of the undersized broadphase; the same checkpoint scores 2292.206 mean return with `gpu_total_aggregate_pairs_capacity = 256*1024`.

### Code

`hang_mug/hang_mug_env_cfg.py` — `RewardsCfg` (verbatim):

```python
@configclass
class RewardsCfg:
    """§6 ladder (iter 0): dense pick-and-place, zero-masked by a one-shot geometric latch, then
    a post-latch hold/release stage, then the terminal success bonus. Composer = sum.

    Declaration order matters: the pre-latch terms read the latch bit BEFORE `on_peg_latch`
    writes it, so the frame the latch fires still pays the dense ladder.
    """

    reach_mug = RewTerm(func=mdp.reach_mug, weight=0.02)
    is_lifted = RewTerm(func=mdp.is_lifted, weight=0.2)
    lift_distance = RewTerm(func=mdp.lift_distance, weight=0.3)
    align_stage = RewTerm(func=mdp.align_stage, weight=2.0)
    thread = RewTerm(func=mdp.thread, weight=6.0)
    on_peg_latch = RewTerm(func=mdp.on_peg_latch, weight=200.0)
    hold_on_peg = RewTerm(func=mdp.hold_on_peg, weight=4.0)
    release_retract = RewTerm(func=mdp.release_retract, weight=6.0)
    success_bonus = RewTerm(func=mdp.success_bonus, weight=2000.0)
```

`hang_mug/mdp/rewards.py` (verbatim, whole file — includes the latch buffers and every helper):

```python
"""§6 reward ladder for the hang_mug task (iter 0).

Ported from the insert-drawer base: a dense pick-and-place ladder that is zero-masked once a
one-shot geometric latch fires, then a post-latch dense stage, then a terminal success bonus
paired with the §4 success `DoneTerm`.

The task-specific changes to the base ladder are the coarse `align_stage` / fine `thread` pair
(a peg is entered end-on, not dropped into from above) and the `hold_on_peg` + `release_retract`
pair that replaces the base's `close_drawer` stage.

All geometry constants and frame helpers come from `geometry.py`; the two goal predicates come
from `terminations.py`. Nothing here re-derives them.

Latch semantics: `on_peg_latch` is the only writer. It fires (+1.0) on the FIRST frame the mug
is geometrically on the peg (settled, held up, hole on the peg axis), then holds the per-(env,
key) bit until the next episode reset (detected via `env.episode_length_buf <= 1`).

Heights are env-local: `env.scene.env_origins[:, 2]` is subtracted from any world z before it is
compared against a threshold, exactly as `terminations.mug_on_peg` does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, FrameTransformer
from isaaclab.utils.math import quat_apply

from .geometry import MUG_GRASP_LOCAL_Z, mug_hole_pos_w, peg_hang_point_w, peg_segment_w
from .terminations import hang_success, mug_on_peg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Module-level per-(env, key) latch buffers.
_LATCH_BUFFERS: dict[tuple, torch.Tensor] = {}


def _get_latch_buffer(env: "ManagerBasedRLEnv", key: str) -> torch.Tensor:
    full_key = (id(env), key)
    if full_key not in _LATCH_BUFFERS:
        _LATCH_BUFFERS[full_key] = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return _LATCH_BUFFERS[full_key]


def _on_peg_latch_active(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Returns float (num_envs,): 1.0 if the on_peg latch fired earlier this episode."""
    latch = _get_latch_buffer(env, "on_peg_once")
    just_reset = env.episode_length_buf <= 1
    return (latch & ~just_reset).float()


def _mug_base_z_env(env: "ManagerBasedRLEnv", mug_name: str = "mug") -> torch.Tensor:
    """Mug root height above the env origin (m)."""
    mug: RigidObject = env.scene[mug_name]
    return mug.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]


def zero_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Constant 0.0 per env — the §6 placeholder, kept for reference."""
    return torch.zeros(env.num_envs, device=env.device)


# ---- Phase 2 — pick + place (zero-masked once the latch fires) ----


def reach_mug(
    env: "ManagerBasedRLEnv",
    std: float = 0.10,
    mug_cfg: SceneEntityCfg = SceneEntityCfg("mug"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """TCP -> mug grasp-point attractor. The grasp point is `MUG_GRASP_LOCAL_Z` up the mug axis,
    not the mug root (which sits on the mug's base plane, i.e. inside the table)."""
    mug: RigidObject = env.scene[mug_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    offset = torch.tensor(
        (0.0, 0.0, MUG_GRASP_LOCAL_Z), dtype=torch.float32, device=env.device
    ).expand(env.num_envs, 3)
    grasp_point_w = mug.data.root_pos_w[:, :3] + quat_apply(mug.data.root_quat_w, offset)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    d = torch.norm(grasp_point_w - ee_w, dim=-1)
    return (1.0 - torch.tanh(d / std)) * (1.0 - _on_peg_latch_active(env))


def is_lifted(
    env: "ManagerBasedRLEnv",
    minimal_height: float = 0.045,
    mug_cfg: SceneEntityCfg = SceneEntityCfg("mug"),
) -> torch.Tensor:
    """Indicator: the mug's base plane is clear of the table (§4 subgoal 3)."""
    return (_mug_base_z_env(env, mug_cfg.name) > minimal_height).float() * (
        1.0 - _on_peg_latch_active(env)
    )


def lift_distance(
    env: "ManagerBasedRLEnv",
    init_z: float = -0.0077,
    target_z: float = 0.1839,
    contact_force_threshold: float = 1e-3,
    mug_cfg: SceneEntityCfg = SceneEntityCfg("mug"),
) -> torch.Tensor:
    """Height ramp on the mug root, gated on BOTH fingertip contact sensors > thr.

    0 on the table (`init_z` = the measured work surface), 1 at the measured mug root height of a
    true hang (`target_z`)."""
    base = ((_mug_base_z_env(env, mug_cfg.name) - init_z) / max(target_z - init_z, 1e-6)).clamp(
        0.0, 1.0
    )
    left: ContactSensor = env.scene["finger_left_contact"]
    right: ContactSensor = env.scene["finger_right_contact"]
    left_f = torch.norm(left.data.force_matrix_w[:, 0, 0, :], dim=-1)
    right_f = torch.norm(right.data.force_matrix_w[:, 0, 0, :], dim=-1)
    grasp_gate = ((left_f > contact_force_threshold) & (right_f > contact_force_threshold)).float()
    return base * grasp_gate * (1.0 - _on_peg_latch_active(env))


def align_stage(
    env: "ManagerBasedRLEnv",
    std: float = 0.10,
    stage_offset: float = 0.035,
    minimal_height_b: float = 0.10,
    mug_cfg: SceneEntityCfg = SceneEntityCfg("mug"),
    rack_name: str = "rack",
) -> torch.Tensor:
    """Coarse attractor: the handle hole -> a staging point just beyond the peg tip, ON the peg's
    line through the hang point. Gated on obstacle clearance (the rack base plate tops out at
    0.035 m)."""
    root, tip = peg_segment_w(env, rack_name)
    axis = tip - root
    u_out = axis / torch.norm(axis, dim=-1, keepdim=True).clamp(min=1e-6)
    stage_point_w = peg_hang_point_w(env, rack_name) + stage_offset * u_out
    d = torch.norm(mug_hole_pos_w(env, mug_cfg.name) - stage_point_w, dim=-1)
    base = 1.0 - torch.tanh(d / std)
    high_enough = (_mug_base_z_env(env, mug_cfg.name) > minimal_height_b).float()
    return high_enough * base


def thread(
    env: "ManagerBasedRLEnv",
    std: float = 0.03,
    minimal_height_b: float = 0.10,
    mug_cfg: SceneEntityCfg = SceneEntityCfg("mug"),
    rack_name: str = "rack",
) -> torch.Tensor:
    """Fine attractor: the handle hole -> the hang point on the peg. Same clearance gate."""
    d = torch.norm(mug_hole_pos_w(env, mug_cfg.name) - peg_hang_point_w(env, rack_name), dim=-1)
    base = 1.0 - torch.tanh(d / std)
    high_enough = (_mug_base_z_env(env, mug_cfg.name) > minimal_height_b).float()
    return high_enough * base


# ---- Phase 3 — one-shot geometric latch ----


def on_peg_latch(
    env: "ManagerBasedRLEnv",
    radial_tol: float = 0.015,
    t_min: float = -0.05,
    t_max: float = 1.0,
    min_height: float = 0.10,
    max_lin_vel: float = 0.05,
    max_ang_vel: float = 1.0,
) -> torch.Tensor:
    """+1.0 on the FIRST frame `mug_on_peg` holds (tightened radial tolerance). Per-env latch;
    the pre-latch dense terms read the bit it writes."""
    latch = _get_latch_buffer(env, "on_peg_once")
    just_reset = env.episode_length_buf <= 1
    latch = torch.where(just_reset, torch.zeros_like(latch), latch)
    now_qualifying = mug_on_peg(
        env,
        radial_tol=radial_tol,
        t_min=t_min,
        t_max=t_max,
        min_height=min_height,
        max_lin_vel=max_lin_vel,
        max_ang_vel=max_ang_vel,
    )
    fire = now_qualifying & (~latch)
    latch = latch | now_qualifying
    _LATCH_BUFFERS[(id(env), "on_peg_once")] = latch
    return fire.float()


# ---- Phase 4 — hold + release ----


def hold_on_peg(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """LIVE `mug_on_peg` indicator (§4's own default params), NOT the sticky latch: dragging the
    mug back off the peg stops the payment."""
    return mug_on_peg(env).float()


def release_retract(
    env: "ManagerBasedRLEnv",
    start_distance: float = 0.09,
    end_distance: float = 0.18,
    mug_cfg: SceneEntityCfg = SceneEntityCfg("mug"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Ramp on ||TCP - mug root||, offset so merely holding the mug (TCP ~0.08 m away) pays 0 and
    a hand 0.18 m away pays 1. Gated on the LIVE `mug_on_peg` predicate."""
    mug: RigidObject = env.scene[mug_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    d = torch.norm(ee_w - mug.data.root_pos_w[:, :3], dim=-1)
    base = ((d - start_distance) / max(end_distance - start_distance, 1e-6)).clamp(0.0, 1.0)
    return base * mug_on_peg(env).float()


# ---- Phase 5 — success bonus ----


def success_bonus(
    env: "ManagerBasedRLEnv",
    radial_tol: float = 0.020,
    min_height: float = 0.10,
    min_ee_distance: float = 0.15,
    contact_force_threshold: float = 0.01,
) -> torch.Tensor:
    """Terminal landmark, identical params to the §4 success `DoneTerm`, so it pays on exactly
    the frame the episode ends."""
    return hang_success(
        env,
        radial_tol=radial_tol,
        min_height=min_height,
        min_ee_distance=min_ee_distance,
        contact_force_threshold=contact_force_threshold,
    ).float()
```

`mdp/rewards.py` depends on `mdp/geometry.py` (§1 code block) and `mdp/terminations.py` (§4 code block). Both are reproduced in full above — nothing in §6 is missing a helper.

### Smoke — §6

Reward decomposition smoke (routes through the training env factory, `harbor/scripts/rl/custom_torch/env_wrapper.py:create_env`, at `num_envs=128`):

```bash
cd <repo>
.venv/bin/python harbor/create-task/isaaclab-franka-hangmug/iter_000/smokes/smoke_s6.py
```

Expected stdout (verdict `status: pass`):

```
S6 OK: 30 steps × 128 envs, reward mean=0.000 std=0.000 terms=9 composer=sum
```

Scene-stability + render smoke (§1–§6 rollout under random actions):

```bash
cd <repo>
.venv/bin/python harbor/create-task/isaaclab-franka-hangmug/smokes/smoke_s6_render.py
```

Expected stdout (verdict `status: pass`):

```
S6 OK: stable for 120 steps across 2 envs; wrote 120 frames -> smoke_s6_render.mp4 (2403.2 KB) + 8 keyframes in smoke_s6_frames (motion=0.94)
```

---

## §7 DR

`<no DR>` — `EventCfg` has **no** `startup` or `interval` terms; all three of its terms are `mode="reset"` (see §3). DR was explicitly not requested for this task (`spec.json:phases.dr_generator = {"status": "skipped", "reason": "DR not requested"}`).

The hook for a future §7 is already wired: `ObservationsCfg.PolicyCfg.__post_init__` sets `enable_corruption = True` while every `Unoise` range is `(0.0, 0.0)`, so observation noise can be turned on by editing ranges only, and `FrankaHangMugEnvCfg_PLAY` already disables corruption for evaluation.

---

