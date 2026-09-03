# task-library — cross-run task design knowledge

Append-only ledgers of reusable task-authoring knowledge, organized by embodiment + task family.
Sibling to the per-agent experience ledgers (`knowledge/experiences/<agent>/*-experience.md`); this tree is
indexed by **what** is being built rather than **which agent** built it.

Use it when authoring a new task (`/harbor:task-create`) or designing reward / DR / observation
for one: read the matching leaf to reuse patterns proven on similar embodiments.

```
task-library/
├── manipulation/                 single- or multi-arm/bimanual pick/place/insert/lift/stack, in-hand
└── locomotion/
    ├── humanoid/                  bipedal humanoid locomotion
    └── quadrupedal/               quadruped locomotion
```

Each leaf is append-only and numbered for stable cross-reference — never renumber existing entries.
Promote a lesson here once it has helped author a second task in that category.

> **Reward weights are NOMINAL.** Every reward spec in this library declares nominal per-step
> `weight` magnitudes — carry them over **as-is** and apply them directly. The declared weight is
> exactly what each term pays per step.

## Upstream code in these entries

A spec is only reproducible if it carries the code verbatim, so entries quote substantial
excerpts of their source benchmark: scene builders, MDP term functions, success predicates and
reward functions, inside fenced code blocks. Those excerpts stay under their upstream license,
not HARBOR's:

| Filename suffix | Upstream | License |
|---|---|---|
| `*-IsaacLab.md` | [Isaac Lab](https://github.com/isaac-sim/IsaacLab) | BSD-3-Clause |
| `*-ManiSkill.md` | [ManiSkill](https://github.com/haosulab/ManiSkill) | Apache-2.0 |

The retained notices are in [`THIRD_PARTY_NOTICES.md`](../../../THIRD_PARTY_NOTICES.md) at the
repository root. When you file a new entry with `/harbor:update-experience`, name the upstream
project and its license in the entry's metadata block if it is not one of the two above — an
excerpt whose provenance is only in the filename is one rename away from being unattributable.
