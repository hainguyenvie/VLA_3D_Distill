# Week 1 Plan — On-Policy 3D / Gaussian Distillation for VLA

## 0. Research target

### Working idea

**On-Policy Spatial Distillation for VLA**

Core hypothesis:

> Privileged 3D supervision is not equally useful everywhere. Its value should become larger on **student-induced / recovery states**, where the policy has deviated from expert demonstrations and 2D observations become spatially ambiguous.

The paper should eventually answer:

\[
\mathbb{E}_{s\sim d_{\pi}}
[\Delta_{3D}(s)]
>
\mathbb{E}_{s\sim d_{E}}
[\Delta_{3D}(s)]
\]

where:

- \(d_E\): expert-state distribution.
- \(d_\pi\): states visited by the current student policy.
- \(\Delta_{3D}(s)\): improvement obtained by adding privileged 3D supervision at state \(s\).

The goal is **not** to claim novelty from simply combining VLA-OPD + GaussianWAM.

The goal is to find and validate a stronger intuition:

> **Teach 3D where the policy's own errors make 3D necessary.**

---

# 1. Week 1 objective

By the end of Week 1, we should have:

1. A reproducible VLA baseline on LIBERO.
2. A reproduced or faithful reimplementation of the essential VLA-OPD loop.
3. A working privileged 3D-supervision pipeline.
4. A controlled comparison between:
   - no 3D;
   - offline/expert-state 3D;
   - on-policy/student-state 3D.
5. Logged failure trajectories and diagnostic statistics.
6. An answer to the first important scientific question:

> **Does 3D supervision actually help more on policy-induced failure states than on expert states?**

If the answer is **no**, do not spend Week 2 polishing the current method. Diagnose why or pivot.

If the answer is **yes**, Week 2 should focus on making the effect selective, efficient, and methodologically novel.

---

# 2. Philosophy for this project

The target is simple:

> **Good intuition + result better than strong baselines.**

Therefore the workflow is:

```text
Reproduce strong baselines
        ↓
Freeze a fair experimental setup
        ↓
Add the smallest version of our idea
        ↓
Measure whether it actually helps
        ↓
Failure analysis / diagnosis
        ↓
Find where and why the gain appears
        ↓
Turn that observation into the real intuition
        ↓
Design the next method component
        ↓
Repeat
```

Do **not** begin by building a complicated final architecture.

The method should be allowed to emerge from experiments.

---

# 3. Main references / competitors

## 3.1 VLA-OPD

**VLA-OPD: Bridging Offline SFT and Online RL for Vision-Language-Action Models via On-Policy Distillation**  
arXiv: 2603.26666

Important pieces to reproduce:

- student-generated trajectories;
- teacher labeling on states visited by the student;
- Reverse-KL style dense distillation;
- OpenVLA-OFT student initialization;
- LIBERO evaluation.

Reported LIBERO numbers in the paper:

| Model | Avg. success |
|---|---:|
| OpenVLA-OFT student init, 1 trajectory | 48.9 |
| VLA-OPD distillation | 87.4 |
| VLA-OPD + GRPO | 93.4 |
| Teacher | 93.9 |

These numbers are **sanity anchors**, not numbers we are allowed to compare against unless our exact setting matches theirs.

---

## 3.2 GaussianWAM

**GaussianWAM: Distilling Geometry and Semantics from 3D Gaussian Fields into World-Action Models**  
arXiv: 2608.24714

Important pieces:

- synchronized multi-view observations;
- geometry + semantic information organized in a Gaussian field;
- semantic/depth/alpha targets rendered from the field;
- supervision applied only during training;
- Gaussian modules removed at inference.

Important reported LIBERO-Plus result on FastWAM:

| Method | LIBERO-Plus overall |
|---|---:|
| FastWAM | 52.05 |
| Direct CLIP + VGGT distillation | 69.37 |
| GaussianWAM | 71.29 |

This tells us something important:

> A large part of the gain comes from privileged semantic/geometric supervision itself, while Gaussian-field organization adds an additional gain.

Therefore we need a **direct 2D/depth teacher baseline**, otherwise we cannot claim the Gaussian field itself matters.

---

## 3.3 FOCAL-VLA

**FOCAL-VLA: Subtask-Guided Geometry Distillation and Implicit World Modeling for Vision-Language-Action Models**  
arXiv: 2609.21228

Important warning:

> "Distill only relevant geometry" is already partially occupied.

FOCAL-VLA uses subtask-relevant image regions for geometry distillation.

Therefore our future selector must eventually be linked to:

- policy-induced failures;
- action disagreement;
- recovery states;
- geometric ambiguity;

rather than only language/subtask relevance.

---

## 3.4 WAM-OPD / related OPD works

These establish that:

> Fresh rollouts from the current student distribution matter.

So **on-policy distillation itself is not the novelty**.

Our claim must be about the interaction between:

\[
\text{policy shift}
\quad\text{and}\quad
\text{value of privileged 3D information}.
\]

---

# 4. Experimental backbone for Week 1

## Primary benchmark

**LIBERO**

Start with:

1. **LIBERO-Object**
2. **LIBERO-Long**

Reason:

- Object gives relatively fast manipulation experiments.
- Long gives error accumulation / recovery states.
- VLA-OPD reports training-efficiency experiments on Object and Long.
- Full four-suite evaluation can be added after the pipeline is stable.

Do not start with RoboTwin or real robot in Week 1.

---

## Student backbone

Preferred:

**OpenVLA-OFT-style student**

Reason:

- closest to the VLA-OPD experimental setting;
- lets us isolate the contribution of our training procedure;
- avoids simultaneously changing backbone + objective + 3D teacher.

---

## Teacher policy

Priority:

1. Official / reproducible VLA-OPD teacher setting if code/checkpoint is available.
2. Otherwise use the strongest reproducible VLA checkpoint trained with substantially more demonstrations.
3. Freeze this teacher for every Week-1 experiment.

The teacher must be **identical across OPD variants**.

Do not change teacher when comparing no-3D vs 3D.

---

# 5. Important experimental principle: test the intuition before building perfect 3DGS

Building a robust estimated 3DGS teacher can consume the entire week.

That is dangerous because the research hypothesis may be wrong.

Therefore use **two levels of privileged 3D**.

---

## Level A — Oracle 3D for hypothesis testing

Use simulator-provided:

- camera intrinsics/extrinsics;
- depth;
- segmentation/object pose if useful.

Construct a clean privileged 3D representation.

Purpose:

> Test whether extra 3D knowledge helps specifically on student-induced states.

This is an **oracle diagnostic**, not the final method.

If even high-quality oracle 3D gives no useful signal, there is little reason to spend time debugging VGGT + Gaussian reconstruction.

---

## Level B — Estimated Gaussian teacher

After the oracle experiment works:

```text
multi-view RGB
    ↓
VGGT / geometry estimator
    ↓
depth + camera
    ↓
Gaussian construction
    ↓
semantic feature binding
    ↓
render semantic / depth / alpha targets
```

This is closer to GaussianWAM and to the final deployable training setup.

---

# 6. Baseline ladder

We need a ladder where every new row adds only one idea.

## B0 — Student Init

OpenVLA-OFT / chosen VLA initialized from the same limited-data setting.

Purpose:

- lower bound;
- verify LIBERO evaluation;
- establish failure trajectories.

---

## B1 — Strong Offline SFT

Same backbone trained with more/full expert demonstrations.

Purpose:

- strong offline baseline;
- approximate upper reference for ordinary imitation learning.

---

## B2 — OPD

Student rollout + frozen teacher action supervision.

```text
student rollout
      ↓
student-visited states
      ↓
teacher action logits
      ↓
Reverse-KL / OPD update
```

Purpose:

- reproduce the key improvement of VLA-OPD;
- make sure our on-policy infrastructure is valid before introducing 3D.

---

## B3 — Offline 3D Distillation

Use privileged 3D supervision on **expert/offline demonstration states only**.

```text
expert states
    ↓
3D / Gaussian targets
    ↓
student auxiliary distillation
```

Purpose:

- reproduce the spirit of GaussianWAM / 3D distillation on our chosen student backbone;
- answer whether 3D helps at all under our setup.

---

## B4 — On-Policy 3D Distillation, all states

Collect student rollouts and add 3D supervision to **every student-visited state**.

```text
student rollout
      ↓
all student states
      ↓
action teacher + 3D teacher
      ↓
student update
```

This is the **naive combination baseline**.

It is important because our future selective method must beat it.

Do not call B4 the final contribution.

---

## M1 — Failure/Disagreement-Triggered On-Policy 3D Distillation

Only query/use expensive 3D supervision on states likely to need it.

Candidate trigger score:

\[
u_t
=
\lambda_1 D_{KL}(\pi_S,\pi_T)
+
\lambda_2 H(\pi_S)
+
\lambda_3 U_{geo}(s_t)
\]

Initial Week-1 implementation can use only:

\[
u_t=D_{KL}(\pi_S,\pi_T)
\]

or teacher-student action disagreement.

If:

\[
u_t>\tau
\]

apply 3D supervision.

Otherwise do ordinary OPD.

Purpose:

- first attempt at turning the intuition into a distinct method;
- test whether selective 3D can match/full 3D performance with fewer 3D teacher queries.

---

# 7. Minimum experiment matrix

Do **not** start with every possible ablation.

First obtain this matrix:

| ID | Student rollout | Action teacher | 3D supervision | State distribution |
|---|---:|---:|---:|---|
| B0 | No | No | No | Expert |
| B1 | No | No | No | Full expert |
| B2 | Yes | Yes | No | Student |
| B3 | No | Optional | Yes | Expert |
| B4 | Yes | Yes | Yes | Student |
| M1 | Yes | Yes | Selective | Student |

The critical comparison is:

\[
B3 \quad vs \quad B4
\]

under as much matched compute as possible.

Scientific question:

> Is the same 3D supervision more useful when trained on student-induced states than on expert states?

---

# 8. Fair-comparison rules

These rules must be frozen before serious experiments.

## Same across compared methods

- backbone;
- pretrained checkpoint;
- tokenizer/action representation;
- training demonstrations;
- task split;
- image resolution;
- observation cameras;
- optimizer;
- learning-rate schedule;
- number of optimization updates;
- batch size / effective batch size;
- rollout budget;
- evaluation seeds;
- number of evaluation episodes;
- teacher policy;
- teacher checkpoint;
- environment version.

---

## When comparing B2 vs B4 vs M1

Keep identical:

- student rollout trajectories where possible;
- number of OPD updates;
- policy teacher calls;
- optimization budget.

Log separately:

- number of 3D teacher calls;
- cost of constructing 3D targets;
- GPU hours.

---

## Random seeds

Workflow:

- smoke test: 1 seed;
- promising experiment: 3 seeds;
- paper result: at least 3 seeds, preferably more if benchmark convention requires.

Do not run 3 seeds before the pipeline works.

---

# 9. Metrics to log from Day 1

Do not log only final success rate.

Every rollout should store:

```text
episode_id
task_id
seed
timestep
success/failure
student action
teacher action/logits
student logits
teacher-student KL
student entropy
camera observations
proprioception
3D-teacher-used?
3D loss
depth / geometry confidence
object visibility if available
distance-to-goal proxies if available
```

---

## Core metrics

### Task success

\[
SR=\frac{\#successful\ episodes}{\#episodes}
\]

Main benchmark metric.

---

### Teacher-student disagreement

\[
D_t=
D_{KL}
(
\pi_S(\cdot|s_t)
\|
\pi_T(\cdot|s_t)
)
\]

Useful for finding hard states.

---

### 3D benefit

For matched conditions:

\[
\Delta_{3D}
=
SR_{\text{with 3D}}
-
SR_{\text{without 3D}}
\]

Compute separately for:

- expert/offline setting;
- student/on-policy setting;
- failure/recovery subsets;
- perturbation types.

---

### 3D query efficiency

\[
QE
=
\frac{SR}{N_{\text{3D-query}}}
\]

Do not necessarily use this exact ratio in the paper, but log both:

- success;
- number/percentage of states using 3D supervision.

Plot:

\[
\text{Success Rate}
\quad vs \quad
\%\text{3D Teacher Queries}.
\]

---

# 10. Failure-analysis taxonomy

Every failed rollout should eventually be assigned one or more labels.

Initial automatic/manual taxonomy:

### F1 — Wrong object / semantic failure

Correct movement but toward wrong target.

### F2 — Spatial localization error

Correct object, wrong relative position/depth.

### F3 — Approach / pose error

Gripper approaches from a bad direction/orientation.

### F4 — Occlusion / visibility failure

Target becomes hidden or partially hidden.

### F5 — Contact / grasp failure

Correct approach but unsuccessful contact/grasp.

### F6 — Recovery failure

Small initial mistake occurs, but policy cannot recover.

### F7 — Long-horizon compounding

Early error creates later impossible state.

### F8 — Camera/viewpoint sensitivity

Action changes incorrectly under viewpoint shift.

### F9 — Layout/object relocation failure

Policy relies on familiar location rather than current geometry.

---

# 11. The main diagnostic we want

For each state/failure bucket \(c\):

\[
\Delta_{3D}^{(c)}
=
Performance_{3D}^{(c)}
-
Performance_{no3D}^{(c)}.
\]

We want to discover a pattern such as:

```text
normal / easy states          +0~2
semantic mistakes             +1
camera shift                  +5
object relocation             +8
occlusion                     +10
recovery states               +12
```

The exact numbers do not matter yet.

The important thing is whether the value of 3D is **highly non-uniform**.

If yes, that becomes the scientific intuition.

---

# 12. Week 1 day-by-day plan

## Day 1 — Environment + evaluation reproduction

### Goal

Make the evaluation pipeline trustworthy before training anything.

### Tasks

- [ ] Create environment for LIBERO.
- [ ] Download/prepare chosen OpenVLA-OFT-style checkpoint.
- [ ] Run inference on a tiny task subset.
- [ ] Verify action scaling / normalization / camera preprocessing.
- [ ] Implement deterministic evaluation script.
- [ ] Save per-episode trajectories.
- [ ] Record package versions and git commits.
- [ ] Add one command that reproduces evaluation from a clean config.

### Deliverable

```text
results/week1/b0_eval.json
logs/week1/day1/
configs/week1/b0_student.yaml
```

### Gate

Do not continue until repeated evaluation with the same seed is stable.

---

## Day 2 — Reproduce Student Init + OPD baseline

### Goal

Obtain a trustworthy no-3D baseline.

### Tasks

- [ ] Reproduce student initialization.
- [ ] Implement student rollout collector.
- [ ] Store full on-policy trajectories.
- [ ] Implement frozen teacher query.
- [ ] Implement Reverse-KL / chosen faithful OPD objective.
- [ ] Run smoke training on a small LIBERO subset.
- [ ] Evaluate B0 vs B2.
- [ ] Plot training curve.
- [ ] Check entropy and teacher-student KL.

### Required plot

```text
x = training updates / environment samples
y = success rate
curves:
    B0 / frozen init
    B2 / OPD
```

### Gate

B2 must clearly improve over B0.

If it does not:

1. debug OPD first;
2. do **not** add 3D yet.

---

## Day 3 — Build 3D diagnostic pipeline

### Goal

Produce privileged 3D targets for a state.

### Phase A: Oracle

- [ ] Read simulator camera intrinsics/extrinsics.
- [ ] Read depth.
- [ ] Back-project to 3D.
- [ ] Fuse available views.
- [ ] Render the reconstructed representation back to each camera.
- [ ] Verify low reprojection error visually/numerically.

### Phase B: Gaussian representation

Each Gaussian can begin minimally as:

\[
g_i=(\mu_i,\Sigma_i,c_i,\alpha_i)
\]

Add semantic feature only after geometry works.

### Cached target candidates

- depth;
- alpha/coverage;
- semantic feature;
- optional object mask.

### Important

First make **depth + coverage** work.

Do not debug CLIP semantics before geometry is correct.

### Deliverable

```text
scripts/cache_3d_teacher.py
src/gaussian_teacher/
cache/week1/...
visualizations/week1/3d_reprojection/
```

---

## Day 4 — Reproduce offline 3D distillation

### Goal

Implement B3 before mixing it with OPD.

### Tasks

- [ ] Decide student layer receiving 3D supervision.
- [ ] Add lightweight prediction heads.
- [ ] Distill depth.
- [ ] Add coverage/alpha if stable.
- [ ] Optionally add semantics.
- [ ] Train on expert demonstration states only.
- [ ] Compare B0/B1 vs B3.

Possible loss:

\[
\mathcal L
=
\mathcal L_{action}
+
\lambda_d\mathcal L_{depth}
+
\lambda_\alpha\mathcal L_{\alpha}
+
\lambda_s\mathcal L_{sem}.
\]

Start with:

\[
\mathcal L
=
\mathcal L_{action}
+
\lambda_d\mathcal L_{depth}.
\]

Add components only if the simple version trains.

### Mandatory baseline

Direct depth/feature distillation **without Gaussian-field organization**.

Reason:

GaussianWAM shows this is already a strong baseline.

We need:

```text
direct 2D/depth supervision
vs
Gaussian-organized supervision
```

eventually.

---

## Day 5 — The critical experiment: expert-state vs student-state 3D

### Goal

Test the central hypothesis.

Run:

### E1 — Offline/expert 3D

\[
s\sim d_E
\]

### E2 — On-policy/student 3D

\[
s\sim d_\pi
\]

Keep:

- same student initialization;
- same 3D loss;
- same amount of optimization;
- matched number of supervised states if possible.

Measure:

\[
\Delta_E
=
SR(B3)-SR(B0/B2\text{ matched})
\]

and

\[
\Delta_\pi
=
SR(B4)-SR(B2).
\]

Main diagnostic:

\[
\Delta_\pi-\Delta_E.
\]

### Desired result

\[
\Delta_\pi>\Delta_E.
\]

Do not hide a negative result.

If:

\[
\Delta_\pi\leq\Delta_E
\]

immediately diagnose why.

---

## Day 6 — Failure analysis

### Goal

Find **where** 3D helps.

Collect trajectories from:

- B0;
- B2;
- B3;
- B4.

### Analyses

#### A. Distance from expert manifold

Create a simple proxy:

\[
d(s,d_E)
\]

using observation embeddings / policy hidden features / nearest-neighbor trajectory distance.

Bin states:

```text
near expert
medium
far from expert
```

Measure 3D gain by bin.

Hypothesis:

\[
\Delta_{3D}
\text{ increases with }
d(s,d_E).
\]

---

#### B. Teacher-student disagreement

Bin by:

\[
D_{KL}(\pi_S,\pi_T).
\]

Measure 3D benefit in each bin.

If benefit is concentrated at high disagreement, we have motivation for selective querying.

---

#### C. Outcome distance

For failed trajectories, separate:

- early states;
- pre-failure states;
- recovery states.

Check whether 3D loss / disagreement spikes before failure.

---

#### D. Visual/spatial categories

Manually inspect at least ~50 failed episodes.

Assign F1–F9 categories from the taxonomy above.

Do not rely only on aggregate SR.

---

## Day 7 — First method + decision day

### Goal

Turn the observed failure pattern into the next method.

Only after Day 6.

### If high-disagreement states benefit most

Implement:

**Disagreement-Triggered 3D OPD**

\[
\mathbf{1}[D_t>\tau]\mathcal L_{3D}.
\]

Sweep:

```text
top 10%
top 25%
top 50%
100%
```

Plot:

\[
SR
\quad vs \quad
3D\ query\ rate.
\]

---

### If recovery states benefit most

Use temporal failure proximity / deviation signal.

Possible direction:

> recovery-aware 3D distillation.

---

### If occlusion states benefit most

Use visibility / Gaussian coverage confidence as the trigger.

Possible direction:

> observability-aware 3D distillation.

---

### If camera/layout shifts benefit most

The paper may move toward:

> distribution-shift-aware 3D post-training.

---

### If there is no meaningful pattern

Do not invent a selector.

Return to diagnosis:

- Is 3D target informative?
- Is the chosen student layer capable of absorbing it?
- Is LIBERO too easy/clean?
- Is the teacher already strong enough that 3D adds nothing?
- Does 3D improve representation but not action?
- Is our reconstruction too noisy?

---

# 13. Go / No-Go gates

## Gate A — Baseline validity

Must have:

- reproducible B0;
- stable evaluation;
- OPD B2 > B0.

Otherwise stop method development.

---

## Gate B — 3D usefulness

At least one of these should occur:

1. B3 > no-3D baseline;
2. B4 > B2;
3. measurable reduction in spatial/recovery failure categories.

If none occur, the chosen 3D supervision is probably not useful.

---

## Gate C — Central intuition

Preferred evidence:

\[
\Delta_\pi > \Delta_E.
\]

Stronger evidence:

\[
\Delta_{3D}
\text{ grows with distance from expert distribution / policy disagreement.}
\]

This is the result that can become the paper intuition.

---

## Gate D — Method potential

Selective method M1 should ideally:

- beat B2;
- match or beat B4;
- use fewer 3D teacher queries than B4.

Ideal story:

```text
B2 OPD                   80
B4 OPD + all-state 3D    86
M1 selective 3D          88
```

while M1 queries 3D on only a subset of states.

Numbers above are illustrative only.

---

# 14. Experiment table to maintain

Create `results/week1_summary.csv` with:

| exp_id | backbone | suite | demos | rollout_budget | opd | 3d | 3d_distribution | selector | 3d_query_rate | seed | SR |
|---|---|---|---:|---:|---|---|---|---|---:|---:|---:|

Never record results only in terminal logs.

---

# 15. Suggested repository structure

```text
project/
├── configs/
│   └── week1/
│       ├── b0_student.yaml
│       ├── b2_opd.yaml
│       ├── b3_offline_3d.yaml
│       ├── b4_onpolicy_3d.yaml
│       └── m1_selective_3d.yaml
│
├── src/
│   ├── policy/
│   ├── distill/
│   │   ├── opd_loss.py
│   │   ├── spatial_loss.py
│   │   └── selector.py
│   ├── rollout/
│   │   ├── collector.py
│   │   └── trajectory_buffer.py
│   └── gaussian_teacher/
│       ├── geometry.py
│       ├── splat.py
│       ├── renderer.py
│       └── targets.py
│
├── scripts/
│   ├── eval_libero.py
│   ├── collect_onpolicy.py
│   ├── train_opd.py
│   ├── cache_3d_teacher.py
│   ├── train_spatial_distill.py
│   └── analyze_failures.py
│
├── results/
│   └── week1/
│
├── trajectories/
│   └── week1/
│
├── visualizations/
│   └── week1/
│
└── notes/
    ├── reproduction_log.md
    └── failure_log.md
```

---

# 16. Reproduction log template

For every reproduced method:

```text
Paper:
Commit:
Environment:
Checkpoint:
Dataset:
Tasks:
Training demos:
Training steps:
Batch size:
Learning rate:
Evaluation episodes:
Seed:
Paper-reported result:
Our result:
Absolute gap:
Known deviations from paper:
Notes:
```

If our setup differs from the paper, write it down immediately.

Do not silently call a result "reproduced".

---

# 17. What NOT to do in Week 1

Do not:

- chase full RoboTwin + LIBERO + real robot simultaneously;
- implement complicated Gaussian semantics before geometry works;
- add RL after OPD before the pure distillation baseline works;
- build a fancy selector before diagnosing where 3D helps;
- tune each baseline with different compute budgets;
- report a "SOTA" gain from a cherry-picked small evaluation subset;
- hide failed runs;
- change backbone midway through the comparison;
- add 5 losses at once.

The first week is for **causal understanding of the method**, not architecture decoration.

---

# 18. Expected Week-1 figures

Even before the final paper, aim to produce these figures.

## Figure A — Baseline reproduction

```text
Success
  ↑
  |                OPD
  |             /
  |          /
  | Student
  +--------------------→ updates
```

---

## Figure B — Where does 3D help?

```text
3D gain
  ↑
  |                    █
  |              █     █
  |        █     █     █
  |  █     █     █     █
  +--------------------------→ distance from expert distribution
     low   medium high  failure
```

This may become the most important intuition figure.

---

## Figure C — Expert vs student-state 3D

```text
                 no 3D      +3D
Expert states      █          █
Student states     █          █████
```

Desired finding:

> 3D adds much more on student-induced states.

---

## Figure D — Performance vs 3D cost

```text
Success
  ↑
  |               * selective
  |             *
  |          *
  |________________________→ % 3D teacher queries
     10   25   50   100
```

---

# 19. End-of-week decision report

At the end of Week 1, write `week1_report.md` answering only these questions:

## Q1. Did we faithfully reproduce the key baselines?

Include:

- reported number;
- reproduced number;
- setup differences.

## Q2. Does privileged 3D supervision improve the chosen VLA?

Yes / no, with numbers.

## Q3. Is the gain larger on student-induced states?

Compare:

\[
\Delta_E
\quad vs \quad
\Delta_\pi.
\]

## Q4. Where does 3D help?

Rank failure categories.

## Q5. Where does 3D not help?

Equally important.

## Q6. Does a simple selective trigger work?

Report:

\[
SR
\quad vs \quad
3D\ query\ rate.
\]

## Q7. What is the strongest intuition supported by evidence?

Do not write the intuition we *wanted*.

Write the intuition the experiments actually support.

## Q8. Should we continue this direction?

Choose one:

```text
GO
GO WITH REFORMULATION
NO-GO / PIVOT
```

---

# 20. Desired result after Week 1

The ideal outcome is **not** merely:

> "Adding Gaussian distillation improves VLA by X%."

That is too close to existing work.

The ideal empirical observation is something like:

> "3D supervision provides little improvement on expert-like states but becomes disproportionately useful as the learned policy deviates from the demonstration manifold, particularly during occlusion and recovery. This suggests that privileged 3D knowledge should be allocated according to policy-induced spatial difficulty rather than uniformly across training samples."

Then the method naturally follows:

\[
\boxed{
\text{On-policy rollout}
\rightarrow
\text{detect spatially difficult states}
\rightarrow
\text{query Gaussian teacher}
\rightarrow
\text{distill correction}
}
\]

That is the direction worth pushing toward a paper.

---

# 21. Immediate first coding order

Start coding in exactly this order:

```text
[1] eval_libero.py
        ↓
[2] trajectory logging
        ↓
[3] student rollout collector
        ↓
[4] teacher query
        ↓
[5] OPD loss
        ↓
[6] reproduce B0 → B2 improvement
        ↓
[7] oracle 3D target generator
        ↓
[8] offline 3D loss
        ↓
[9] B3 experiment
        ↓
[10] on-policy 3D cache / generation
        ↓
[11] B4 experiment
        ↓
[12] failure analysis
        ↓
[13] only then implement M1 selector
```

**Do not skip directly to step 13.**
