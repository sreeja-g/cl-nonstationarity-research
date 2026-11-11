# Research history and supervisor correspondence

> One file, covering everything that happened in this project before the 2026-11-07 reset: the full research design history (why it changed, what it ended up as) and a summary of every email exchanged with the supervisor. Written so that someone coming back after a break can get the whole backstory from this file alone, without having to open the archived hub. The archived hub (`phase-1-archive/research-hub/`, not on GitHub) still has every file this summarizes, verbatim, if a detail here needs checking against the source.

---

## 1. What this project is

The study asks: when a model learns a sequence of visual tasks one after another, does the size of the **distribution shift** between consecutive tasks predict how much it **forgets**? And does starting from a pretrained model (instead of training from scratch) make it less sensitive to that shift?

It's being done as part of a master's research project, supervised by **Dr. Nigel Basta**, and run on a self-funded RunPod GPU budget of about $50.

---

## 2. Timeline

| Date | What happened |
|---|---|
| 2026-10-05 | Original research log created: notation, shift taxonomy, theoretical write-up, no code yet. |
| 2026-10-06 | Two internal adversarial reviews applied to the theory write-up (~25 findings fixed). |
| 2026-10-08 | Docs restructured into the numbered `research-hub/` format. |
| 2026-10-09 | **Supervisor's first substantive email** on the original ("v1") proposal — 25 suggestions (S1–S25). See [§4.1](#email-2026-10-09). |
| 2026-10-11 | **Study redesigned ("v2")** in direct response: two sharp hypotheses, same-architecture scratch-vs-pretrained arms replacing the ResNet-vs-ViT design, simplified method list. Sent to the supervisor the same day. See [§4.2](#reply-2026-10-11). Code for v2 built on branch `refactor/nonstationarity-v2` (never run on a GPU at this point). |
| 2026-10-12 | **Supervisor approves v2**, asks one follow-up question about tuning. See [§4.3](#email-2026-10-12). Two rounds of internal code/hub review fix ~20 more findings, including a real statistical bug (see [§3.6](#stats-bug)). |
| 2026-10-12 | Researcher replies with the tuning protocol and asks two clarifying questions (grid vs. Optuna; tune once vs. per-benchmark). See [§4.4](#reply-2026-10-12). |
| 2026-10-12 | **Supervisor answers the tuning questions** (tune per-benchmark, use Optuna) and offers university GPU cluster access. See [§4.5](#email-2026-10-12-2). |
| 2026-11-07 | **Full workflow reset.** Everything above (all hub docs + all code) archived into `phase-1-archive/`, untouched. A fresh, much simpler hub and code repo started, to be built one file at a time. This file is written as part of that reset, to preserve the history above in one place. |

Nothing was ever run on a real GPU before the reset — the code existed (236 passing tests on CPU) but the actual training runs, the pilot, and the full grid had not started.

---

## 3. The research design, as it stood (v2, the final version before the reset)

### 3.1 The two hypotheses

- **H1** — When the shift between two tasks is larger, regularization-based methods (EWC, LwF) forget more, while replay-based methods (ER, DER++) are less affected.
- **H2** — A pretrained, partly-frozen model is less affected by shift than the same model trained from scratch.

### 3.2 Models ("arms")

Same network, scratch vs. pretrained, so pretraining is isolated from architecture:

| Arm | Architecture | Init | What's trainable | Runs on |
|---|---|---|---|---|
| `r18_scratch` | ResNet-18 | scratch | everything | all 4 benchmarks |
| `r18_places` | ResNet-18 | Places365 | last block + classifier only | all 4 benchmarks |
| `r50_scratch` | ResNet-50 | scratch | everything | all 4 benchmarks |
| `r50_places` | ResNet-50 | Places365 | last block + classifier only | all 4 benchmarks |
| `r18_imagenet` | ResNet-18 | ImageNet | last block + classifier only | CORe50 only (extra check) |
| `mbv2_imagenet` | MobileNetV2 | ImageNet | last block + classifier only | CORe50 only (extra check) |
| `r18_places_all` (optional) | ResNet-18 | Places365 | everything | H2 confound check, pilot benchmark only |

Pretraining uses **Places365**, not ImageNet, specifically to avoid the pretrained model having already seen ImageNet-100's classes (which would make that comparison meaningless).

### 3.3 Four benchmarks

| Benchmark | Shift type | Notes |
|---|---|---|
| ImageNet-100 | class-incremental | split into 20 tasks of 5 classes |
| CLEAR-10 | gradual temporal drift | ~9 usable transitions — results shown but not used for firm conclusions |
| DomainNet | sudden style shift | few tasks (6 domains) — same caveat as CLEAR-10 |
| CORe50 (NIC protocol) | mixed | ~40 usable consecutive transitions |

### 3.4 Five methods (unchanged across every arm, all built into the Avalanche library)

Naive fine-tuning (baseline), EWC, LwF (regularization family), ER, DER++ (replay family).

### 3.5 Measuring shift

Two numbers per task transition:
- **MMD** on frozen-encoder features (two encoders used — a supervised ImageNet ResNet-50 and a self-supervised DINO ResNet-50 — as a robustness check against relying on just one).
- **CKA** on a fixed, held-out probe set of old-task images, to see how much the model's internal representation of old tasks drifts.

### 3.6 The analysis — and a real statistical bug that got caught <a id="stats-bug"></a>

The original plan (per-transition pooled regression with a clustered bootstrap) was replaced on 2026-10-12 after an internal review found it would falsely reject a true "no effect" result 35–50% of the time, because transitions are shared across methods but each method's sensitivity to shift is not. The fix: a **two-stage test** — fit one shift-vs-forgetting slope per individual run first (stage 1), then compare those slopes across method families or across scratch/pretrained pairs (stage 2). This is the version that was still current at the reset.

### 3.7 Budget and scope

~$50 total on a RunPod A6000 GPU ($0.59/h), capped at 40 GPU-hours by default. If time or money runs short, the rule is to cut **methods** before cutting **benchmarks**, since the benchmarks are what provide the shift variation the whole study depends on.

### 3.8 What was still unresolved at the reset

A non-exhaustive list of the open items that mattered most (the full list — 16 open questions — is archived in `phase-1-archive/research-hub/02-study-design/open-questions.md`):

- **Disk space.** The planned 30GB RunPod volume is too small even for the pilot, let alone the full ImageNet download.
- **Tuning protocol conflict.** Decision D15 (tune once, reuse everywhere) was never formally updated after the supervisor's 2026-10-12 answer (tune per-benchmark with Optuna) — the two documents disagreed with each other at the time of the reset.
- **Epoch budget** was still a placeholder (`epochs: 1`).
- Several checkpoint hashes (DINO encoder, ImageNet weights) weren't pinned yet.
- Whether to run the optional confound-check arm (`r18_places_all`) was undecided, pending the pilot's measured GPU cost.

---

## 4. Supervisor correspondence, summarized

Every email is archived verbatim in `phase-1-archive/research-hub/06-supervisor-feedback/` (not on GitHub — that folder has not been pushed anywhere). This section is the plain-language summary of each one, in order.

<a id="email-2026-10-09"></a>

### 4.1 Supervisor's email — 2026-10-09 (on the original "v1" proposal)

Encouraging but pointed. Main asks:
- Tighten the vague goal ("shift helps explain forgetting") into one or two sharp, falsifiable hypotheses, decided *before* running anything.
- The four benchmarks differ in more than just shift type (size, task count, resolution, difficulty) — these are confounds that need controlling for or at least reporting.
- Be deliberate about which forgetting-prevention method pairs with which architecture (this was about the original L2P/PackNet/ViT design, later dropped entirely).
- Think hard about the frozen-ViT arm: it might show so little forgetting everywhere that there's nothing to compare.
- MMD encoder choice matters and needs a robustness check; CKA probe set needs to be fixed and comparable across methods.
- Run multiple seeds, use a pilot before scaling up, and be careful about tuning so it doesn't leak future information.
- Caution against claiming causation — shift and forgetting can only be shown to be *associated*.

This email is what triggered the full v1 → v2 redesign.

<a id="reply-2026-10-11"></a>

### 4.2 Researcher's reply — 2026-10-11 (the v2 redesign)

Sreeja's response, sent the same day as the redesign. Key points: two sharp hypotheses (H1, H2, as in [§3.1](#31-the-two-hypotheses)); kept all four benchmarks but added a confounds table and measured shift at every task transition instead of once per benchmark; replaced the ResNet-vs-ViT comparison with same-architecture scratch-vs-Places365-pretrained pairs (ResNet-18 and ResNet-50); dropped L2P and PackNet since there's no ViT anymore; MMD with a second encoder as a robustness check; CKA on a fixed set of old-task images; 3 seeds with tuning only on first tasks; associational language only. Also asked whether a university GPU cluster might be available instead of RunPod.

<a id="email-2026-10-12"></a>

### 4.3 Supervisor's email — 2026-10-12 (approves v2, one follow-up question)

Very positive — called the revised hypotheses concrete and testable, approved the new model comparison and the simplified method list, approved the shift/CKA measurement plan. Two specific notes: (1) make sure the CKA probe images are held out from training on *all* tasks, not just old ones, or the measurement gets confounded by memorization; (2) the shift-forgetting relationship might not be linear, and might vary unpredictably by benchmark — a mixed result is still a valid finding, not a failure. Confirmed the university GPU cluster (A100/V100) is available on request, and that $50 on RunPod is also a reasonable budget if that's preferred. Asked one follow-up question: how exactly will hyperparameter tuning work in the pilot (grid search on first task vs. literature values vs. something else)?

<a id="reply-2026-10-12"></a>

### 4.4 Researcher's reply — 2026-10-12 (tuning protocol, two questions)

Accepted the offer of university cluster access. Described the tuning plan: tune on first tasks only, validation split carved from training data, search ranges from the original papers, scratch and pretrained models tuned separately with the same ranges and same number of trials. Asked two open questions: (1) tune once on the pilot benchmark and reuse everywhere, or tune separately per benchmark? (2) simple grid search, or Optuna (would make it easy to give every method an equal number of trials)?

<a id="email-2026-10-12-2"></a>

### 4.5 Supervisor's email — 2026-10-12, second (answers the tuning questions)

Will introduce the researcher to the systems administrator for cluster access that week; recommended starting with a short test job and writing a job script that takes seed/benchmark/method as arguments. On tuning: **tune each benchmark separately on its own first tasks**, not once on a pilot and reused — the benchmarks differ too much (resolution, task size, class count) for one tuning to be fair. On search method: **recommended Optuna** over a grid, with the same fixed trial budget for every method (sized for the most complex method, DER++), a log-scale search range of 10x in each direction from the paper's value, a fixed sampler seed for reproducibility, the paper value itself as the first trial, and every trial recorded (not just the best). Keep tuning transitions out of the final shift/forgetting analysis, or flag them clearly. Recommended reading van de Ven & Tolias's "three scenarios for continual learning" paper to sharpen how the four benchmarks are framed.

**Note:** this answer directly conflicts with decision D15 in the old design log (which said "tune once, reuse everywhere") — that conflict was flagged in the hub but never formally resolved before the reset. If the project picks back up on hyperparameter tuning, this is the first thing to reconcile.

---

## 5. Where to look for more detail

Everything summarized above exists in much greater (and more technical) detail in the archived hub, at `phase-1-archive/research-hub/`:
- `02-study-design/decision-log.md` — every individual design decision, with status and rationale.
- `02-study-design/open-questions.md` — the full list of 16 open questions at the time of the reset.
- `05-journal/research-journal.md` — the dated, append-only research log with full adversarial-review critiques.
- `06-supervisor-feedback/` — every email, verbatim.

That folder is local-only (not pushed anywhere) and was left untouched by the 2026-11-07 reset.
