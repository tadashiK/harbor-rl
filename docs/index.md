---
layout: home

hero:
  name: HARBOR
  text: Robot RL, as a request
  tagline: Point it at a simulator. Describe a task. Get a trained policy — with an executable check at every step.
  image:
    src: /logo.svg
    alt: HARBOR
  actions:
    - theme: brand
      text: Get started
      link: /guide/
    - theme: alt
      text: How it works
      link: /guide/harness
    - theme: alt
      text: Read the paper
      link: https://arxiv.org/abs/2606.08610
---

<div style="max-width: 980px; margin: 4rem auto 0; text-align: center;">

## One prompt, End-to-end workflow

Set up simulation · Write the task · Design the reward · Wire the algorithms · Train the policy

<!-- preload=metadata so the file is not pulled down by every visitor who never presses
     play; the poster carries the frame until they do. -->
<video src="/demo.mp4" controls playsinline preload="metadata"
       poster="/walkthrough-poster.webp"
       style="width: 100%; border-radius: 12px;"></video>

## One harness, different tasks · robots · simulators

<img src="/gallery.webp" alt="Every HARBOR task rendered across IsaacLab, ManiSkill and Genesis" style="width: 100%; border-radius: 12px;">

</div>

<div style="max-width: 720px; margin: 3rem auto 0;">

## Why a harness

Reinforcement learning works. The pipeline around it is what costs weeks.

Practitioners build the task, shape the reward, calibrate randomization, tune hyperparameters — and pay that cost again for every new simulator, task, and algorithm. Prior automation targets one stage at a time, so the integration burden between stages is repeatedly re-paid by hand.

HARBOR treats the whole problem as **harness engineering**: shifting human effort from executing each step to designing an agent-readable workflow with verifiable interfaces. Robot RL is unusually well suited to this, because the MDP already exposes stable interfaces — state, action, reward, dynamics, termination — and simulators already produce executable feedback.

HARBOR cannot prove your policy is semantically correct. What it does is turn the common RL engineering failures into **gate failures that surface before they propagate downstream**.

<div class="harbor-cards">
  <div class="harbor-card">
    <h3>Long-horizon automation</h3>
    <p>A reliable workflow over tightly coupled decisions — related work automates individual steps.</p>
  </div>
  <div class="harbor-card">
    <h3>Wall-clock efficiency</h3>
    <p>The iterative stages — reward engineering and hyperparameter tuning — are where the clock goes.</p>
  </div>
  <div class="harbor-card">
    <h3>Self-improvement</h3>
    <p>Human heuristics and existing examples, carried into the next run by in-context learning.</p>
  </div>
  <div class="harbor-card">
    <h3>Interpretability and controllability</h3>
    <p>Full trace documentation at every stage, and human intervention wherever you want it.</p>
  </div>
</div>

[Read the concepts →](/guide/harness)

</div>
