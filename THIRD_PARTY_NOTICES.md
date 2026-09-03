# Third-party notices

HARBOR is licensed under Apache 2.0 (see [`LICENSE`](LICENSE)). Parts of it are derived from,
or reproduce portions of, the projects below. Their notices are retained here as their licenses
require, and each derived file carries a short pointer back to this document.

Nothing here restricts your use of HARBOR beyond what these upstream licenses already ask for:
keep the notice with the code.

---

## 1. PQL — `knowledge/templates/rl-integration-generator/custom_torch/`

**Upstream:** <https://github.com/Improbable-AI/pql> · MIT

The `custom_torch` algorithm tree that HARBOR renders into a benchmark repository —
`algo/`, `models/`, `replay/` and `utils/` — is adapted from PQL. The equivariant-policy
imports, the multi-agent code path, and the `info_track_keys` plumbing were stripped, and
per-term reward and success logging were made optional; the algorithmic content of PPO, SAC,
the replay buffers and the tensor utilities is PQL's. The `scripts/` subtree
(`train.py`, `eval.py`, `render.py`, `env_wrapper.py`) and `algo/td3.py` are HARBOR's own.

```
MIT License

Copyright (c) 2023 Improbable AI Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 2. LocoMuJoCo — `knowledge/templates/rl-integration-generator/custom_jax/`

**Upstream:** <https://github.com/robfiras/loco-mujoco> · MIT

The `custom_jax` tree ports the algorithmic recipe of
`loco_mujoco/algorithms/ppo_jax.py` — the PPO update, the optimizer construction, the
`LogWrapper`/`VecEnv` wrapper stack and the episode-completion callback — onto HARBOR's own
config schema and checkpoint format. `custom_jax/scripts/env_wrapper.py` additionally builds
LocoMuJoCo MJX environments through that project's public `RLFactory` API.

```
MIT License

Copyright (c) 2024 Al-Hafez

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 3. Isaac Lab — `knowledge/experiences/task-library/**-IsaacLab.md`

**Upstream:** <https://github.com/isaac-sim/IsaacLab> · BSD-3-Clause

The task library records each task as a reproduction spec, and a spec is only reproducible if
it carries the code verbatim. Entries whose filename ends in `-IsaacLab` therefore quote
substantial excerpts of Isaac Lab task source — environment configs, MDP term functions, scene
and event definitions — inside fenced code blocks. Those excerpts remain under BSD-3-Clause.

```
Copyright (c) 2022-2025, The Isaac Lab Project Developers
(https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).

All rights reserved.

SPDX-License-Identifier: BSD-3-Clause

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

---

## 4. ManiSkill — `knowledge/experiences/task-library/**-ManiSkill.md`

**Upstream:** <https://github.com/haosulab/ManiSkill> · Apache-2.0

As above, for entries whose filename ends in `-ManiSkill`: they quote substantial excerpts of
ManiSkill task source — `_load_scene`, `_initialize_episode`, `evaluate`,
`compute_dense_reward` and the surrounding environment classes. Those excerpts remain under the
Apache License 2.0, whose full text is the same one HARBOR itself ships as [`LICENSE`](LICENSE).

Where an excerpt was modified in the course of adapting a task, the surrounding prose in the
entry says so; the excerpts themselves are reproduced as found upstream.

---

## Gallery and demo media

`assets/gallery/*` and `docs/public/*` are renders produced by running HARBOR against the
simulators named in each filename. The robot models and scene assets visible in those renders
are the property of their respective upstream projects and are shown here as output of those
simulators, not redistributed as assets.
