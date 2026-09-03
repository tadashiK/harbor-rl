# Task library

Every task HARBOR has authored or reproduced, kept as a self-contained implementation spec — scene, actions, reset, termination, observation, reward and DR, with verbatim code. Together they are what a new task is adapted *from*: authoring runs a search over this library first, so a new task starts from the closest prior one rather than from a blank file.

Each entry is reproducible with [`/harbor:task-create`](/guide/commands) `from=<spec>`.

<details>
<summary><b>Manipulation</b></summary>

Arms and hands acting on objects — reaching, grasping, placing, inserting, stacking, and in-hand reorientation.

| Task | Robot | Simulator | What it does |
|---|---|---|---|
| [`dexsuite-reorient-kuka-allegro-IsaacLab`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/manipulation/dexsuite-reorient-kuka-allegro-IsaacLab.md) | Kuka LBR iiwa7 arm + Allegro hand (23 DoF) | IsaacLab | Reorient a randomly shaped object to a commanded 6-DoF goal pose. |
| [`dexterous-grasp-IsaacLab`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/manipulation/dexterous-grasp-IsaacLab.md) | UFactory UF850 arm + Allegro right hand (22 DoF) | IsaacLab | Grasp a small rigid object off a table and lift it to a target height. |
| [`g1-bimanual-place-apple-ManiSkill`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/manipulation/g1-bimanual-place-apple-ManiSkill.md) | Unitree G1 humanoid, simplified upper body, bimanual (25 DoF, fixed base) | ManiSkill | Pick an apple off a kitchen counter and place it in a bowl. |
| [`g1-bimanual-transport-box-ManiSkill`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/manipulation/g1-bimanual-transport-box-ManiSkill.md) | Unitree G1 humanoid, simplified upper body + head camera, bimanual (25 DoF, fixed base) | ManiSkill | Lift a box from one table and carry it across to another. |
| [`handover-IsaacLab`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/manipulation/handover-IsaacLab.md) | Bimanual UF850 arms + dual Allegro hands (44 DoF) | IsaacLab | Grasp a bottle with one hand, lift it to a mid-air handover pose, and transfer it to the other hand. |
| [`hang-mug-on-rack-IsaacLab`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/manipulation/hang-mug-on-rack-IsaacLab.md) | Franka Emika Panda (7 DoF arm + parallel gripper) | IsaacLab | Pick up a mug and hang it by its handle on a peg rack. |
| [`inhand-rotate-object-ManiSkill`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/manipulation/inhand-rotate-object-ManiSkill.md) | Allegro right hand with FSR touch links (16 DoF, fixed base) | ManiSkill | Rotate an object in-hand to a target orientation without dropping it. |
| [`insert-drawer-IsaacLab`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/manipulation/insert-drawer-IsaacLab.md) | Franka FR3 arm + Franka hand (single arm) | IsaacLab | Pick a cube off the table, place it inside an open drawer, and push the drawer closed. |
| [`insert-drawer-bimanual-allegro-IsaacLab`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/manipulation/insert-drawer-bimanual-allegro-IsaacLab.md) | Bimanual UF850 arms + dual Allegro hands (44 DoF) | IsaacLab | Place an object into a drawer using two dexterous arms. |
| [`lift-box-IsaacLab-v2`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/manipulation/lift-box-IsaacLab-v2.md) | Two Franka FR3 arms + Franka hands (dual-arm cooperative) | IsaacLab | Two arms cooperatively grasp a eurobox and lift it off the table. |
| [`lift-box-IsaacLab`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/manipulation/lift-box-IsaacLab.md) | Two Franka FR3 arms + Franka hands (dual-arm cooperative) | IsaacLab | Two arms cooperatively grasp a eurobox and lift it off the table. |
| [`lift-box-bimanual-allegro-IsaacLab`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/manipulation/lift-box-bimanual-allegro-IsaacLab.md) | Bimanual UF850 arms + dual Allegro hands (44 DoF) | IsaacLab | Two dexterous hands cooperatively grasp a tote box and lift it to a commanded pose. |
| [`lift-cube-franka-IsaacLab`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/manipulation/lift-cube-franka-IsaacLab.md) | Franka Emika Panda (7 DoF arm + parallel gripper) | IsaacLab | Reach a cube, lift it clear of the table, and carry it to a commanded goal pose. |
| [`lift-peg-upright-ManiSkill`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/manipulation/lift-peg-upright-ManiSkill.md) | Franka Panda (default; Fetch also supported) | ManiSkill | Move a peg lying flat on the table into an upright orientation. |
| [`open-cabinet-door-fetch-ManiSkill`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/manipulation/open-cabinet-door-fetch-ManiSkill.md) | Fetch mobile manipulator | ManiSkill | Drive to a cabinet and open its door at least three quarters of the way. |
| [`open-drawer-franka-IsaacLab`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/manipulation/open-drawer-franka-IsaacLab.md) | Franka Emika Panda (7 DoF arm + parallel gripper) | IsaacLab | Grasp a cabinet drawer handle and pull the drawer open. |
| [`peg-insertion-side-ManiSkill`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/manipulation/peg-insertion-side-ManiSkill.md) | Franka Panda with wrist camera (`panda_wristcam`) | ManiSkill | Insert a peg sideways into a matching hole in a box. |
| [`pick-cube-ManiSkill`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/manipulation/pick-cube-ManiSkill.md) | Franka Panda (default; Fetch / xArm6-Robotiq / SO100 / WidowXAI also supported) | ManiSkill | Grasp a cube and move it to a goal position marked by a sphere. |
| [`pick-place-object-IsaacLab`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/manipulation/pick-place-object-IsaacLab.md) | Bimanual UF850 arms + dual Allegro hands (44 DoF) | IsaacLab | Two hands pick two cubes off a table and drop them into a tote, in a fixed order. |
| [`pick-ycb-object-ManiSkill`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/manipulation/pick-ycb-object-ManiSkill.md) | Franka Panda with wrist camera (`panda_wristcam`; Panda / Fetch also supported) | ManiSkill | Lift a randomly sampled YCB object and move it to a goal position. |
| [`pickplace-gr1t2-bimanual-IsaacLab`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/manipulation/pickplace-gr1t2-bimanual-IsaacLab.md) | Fourier GR1T2 bimanual humanoid (two 7-DoF arms + dexterous hands) | IsaacLab | A bimanual humanoid picks a steering wheel off a packing table and places it. |
| [`place-sphere-in-bin-ManiSkill`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/manipulation/place-sphere-in-bin-ManiSkill.md) | Franka Panda (default; Fetch also supported) | ManiSkill | Pick up a sphere and place it inside a bin. |
| [`plug-charger-ManiSkill`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/manipulation/plug-charger-ManiSkill.md) | Franka Panda with wrist camera (`panda_wristcam`) | ManiSkill | Grasp a charger and plug it into a wall receptacle. |
| [`poke-cube-ManiSkill`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/manipulation/poke-cube-ManiSkill.md) | Franka Panda (default; Fetch also supported) | ManiSkill | Use a peg to poke a cube to a goal position. |
| [`pull-cube-ManiSkill`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/manipulation/pull-cube-ManiSkill.md) | Franka Panda (default; Fetch also supported) | ManiSkill | Pull a cube across the table toward the robot into a goal region. |
| [`pull-cube-with-tool-ManiSkill`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/manipulation/pull-cube-with-tool-ManiSkill.md) | Franka Panda (default; Fetch also supported) | ManiSkill | Use an L-shaped tool to pull an out-of-reach cube into reach. |
| [`push-block-franka-IsaacLab`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/manipulation/push-block-franka-IsaacLab.md) | Franka Emika Panda (gripper forced closed, used as a flat pusher) | IsaacLab | Push a block across the table to a target marker using a closed gripper as a pusher. |
| [`push-cube-ManiSkill`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/manipulation/push-cube-ManiSkill.md) | Franka Panda (default; Fetch also supported) | ManiSkill | Push a cube across the table into a goal region. |
| [`push-t-shape-ManiSkill`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/manipulation/push-t-shape-ManiSkill.md) | Franka Panda with stick end-effector (`panda_stick`) | ManiSkill | Push a T-shaped block into alignment with a target T outline. |
| [`reach-franka-IsaacLab`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/manipulation/reach-franka-IsaacLab.md) | Franka Emika Panda (7 DoF arm, no gripper action) | IsaacLab | Move the end-effector to a commanded goal pose. |
| [`reorient-cube-allegro-IsaacLab`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/manipulation/reorient-cube-allegro-IsaacLab.md) | Allegro hand (16 DoF, fixed in air, no arm) | IsaacLab | Rotate a cube in-hand to match a commanded goal orientation. |
| [`roll-ball-to-goal-ManiSkill`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/manipulation/roll-ball-to-goal-ManiSkill.md) | Franka Panda | ManiSkill | Roll a ball across the table into a goal region. |
| [`rotate-valve-dclaw-ManiSkill`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/manipulation/rotate-valve-dclaw-ManiSkill.md) | D'Claw three-finger hand (9 DoF, fixed base) | ManiSkill | Rotate a three-spoke valve with a three-finger hand. |
| [`stack-three-cube-IsaacLab`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/manipulation/stack-three-cube-IsaacLab.md) | Franka FR3 arm + Franka hand (single arm) | IsaacLab | Stack three cubes into a tower, in order. |
| [`stack-three-cube-pyramid-ManiSkill`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/manipulation/stack-three-cube-pyramid-ManiSkill.md) | Franka Panda with wrist camera (`panda_wristcam`; Panda / Fetch also supported) | ManiSkill | Arrange three cubes into a pyramid. |
| [`stack-two-cube-ManiSkill`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/manipulation/stack-two-cube-ManiSkill.md) | Franka Panda with wrist camera (`panda_wristcam`; Panda / Fetch also supported) | ManiSkill | Stack one cube on top of another. |
| [`stir-bowl-IsaacLab`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/manipulation/stir-bowl-IsaacLab.md) | Bimanual UF850 arms + dual Allegro hands (44 DoF) | IsaacLab | One hand holds an egg-beater and stirs the contents of a bowl. |
| [`threading-IsaacLab`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/manipulation/threading-IsaacLab.md) | Bimanual UF850 arms + dual Allegro hands (44 DoF) | IsaacLab | Thread a drill head through the hole of a cube held by the other hand. |
| [`trifinger-rotate-cube-ManiSkill`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/manipulation/trifinger-rotate-cube-ManiSkill.md) | TriFingerPro three-finger hand (9 DoF, fixed base) | ManiSkill | Rotate a cube to a target orientation with a three-finger manipulator. |
| [`two-arm-pick-cube-ManiSkill`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/manipulation/two-arm-pick-cube-ManiSkill.md) | Two Franka Panda arms with wrist cameras (`panda_wristcam` x2, multi-agent) | ManiSkill | Two arms cooperate to pick a cube and move it to a goal sphere. |
| [`two-arm-stack-cube-ManiSkill`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/manipulation/two-arm-stack-cube-ManiSkill.md) | Two Franka Panda arms with wrist cameras (`panda_wristcam` x2, multi-agent) | ManiSkill | Two arms cooperate to stack one cube on another at a target. |

</details>

<details>
<summary><b>Humanoid locomotion</b></summary>

Bipedal whole-body control: standing, velocity tracking, and footstep following.

| Task | Robot | Simulator | What it does |
|---|---|---|---|
| [`footstep-g1-IsaacLab`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/locomotion/humanoid/footstep-g1-IsaacLab.md) | Unitree G1 bipedal humanoid (37 DoF, hand-equipped) | IsaacLab | Follow an alternating sequence of swing-foot touchdown poses on flat ground. |
| [`g1-humanoid-stand-ManiSkill`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/locomotion/humanoid/g1-humanoid-stand-ManiSkill.md) | Unitree G1 humanoid, simplified-legs URDF (37 DoF, floating base) | ManiSkill | Stand upright and stay balanced on a flat plane. |
| [`h1-humanoid-stand-ManiSkill`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/locomotion/humanoid/h1-humanoid-stand-ManiSkill.md) | Unitree H1 humanoid, simplified (19 DoF, floating base) | ManiSkill | Stand upright and stay balanced on a flat plane. |
| [`walk-g1-flat-IsaacLab`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/locomotion/humanoid/walk-g1-flat-IsaacLab.md) | Unitree G1 bipedal humanoid (37 DoF, hand-equipped) | IsaacLab | Track a commanded base velocity while walking on flat ground. |
| [`walk-h1-flat-IsaacLab`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/locomotion/humanoid/walk-h1-flat-IsaacLab.md) | Unitree H1 bipedal humanoid (19 DoF) | IsaacLab | Track a commanded base velocity while walking on flat ground. |
| [`walk-h1-rough-IsaacLab`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/locomotion/humanoid/walk-h1-rough-IsaacLab.md) | Unitree H1 bipedal humanoid (19 DoF) | IsaacLab | Track a commanded base velocity across procedurally generated rough terrain. |

</details>

<details>
<summary><b>Quadrupedal locomotion</b></summary>

Four-legged velocity tracking and goal-reaching over flat and rough terrain.

| Task | Robot | Simulator | What it does |
|---|---|---|---|
| [`anymal-reach-goal-ManiSkill`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/locomotion/quadrupedal/anymal-reach-goal-ManiSkill.md) | ANYbotics ANYmal-C quadruped (12 DoF) | ManiSkill | Walk to a commanded goal position on flat ground. |
| [`anymal-spin-ManiSkill`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/locomotion/quadrupedal/anymal-spin-ManiSkill.md) | ANYbotics ANYmal-C quadruped (12 DoF) | ManiSkill | Spin in place about the vertical axis as fast as possible without falling. |
| [`walk-anymal-flat-IsaacLab`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/locomotion/quadrupedal/walk-anymal-flat-IsaacLab.md) | ANYbotics ANYmal-C quadruped (12 DoF) | IsaacLab | Track a commanded base velocity while trotting on flat ground. |
| [`walk-anymal-rough-IsaacLab`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/locomotion/quadrupedal/walk-anymal-rough-IsaacLab.md) | ANYbotics ANYmal-C quadruped (12 DoF) | IsaacLab | Track a commanded base velocity across procedurally generated rough terrain. |
| [`walk-spot-flat-IsaacLab`](https://github.com/supersglzc/harbor-rl/blob/main/knowledge/experiences/task-library/locomotion/quadrupedal/walk-spot-flat-IsaacLab.md) | Boston Dynamics Spot quadruped (12 DoF, remotized-PD knee) | IsaacLab | Track a commanded base velocity while walking on flat ground. |

</details>
