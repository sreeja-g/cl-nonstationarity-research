# File map — what we're going to build, and in what order

> This is the concrete build list. Part A is fully planned out, file by file, since we're starting there. Part B is a rough first pass — we'll firm it up once Part A is done and we've seen what the data actually looks like. All code lives in the `cl-nonstationarity-research/` folder, not in `research-hub/` (docs and code stay separate).

## Folder structure (to be built as we go — nothing here exists yet)

```
cl-nonstationarity-research/
├── part-a-distribution-checks/
│   ├── 01_setup_check.ipynb
│   ├── 02_explore_imagenet100.ipynb
│   ├── 03_explore_clear10.ipynb
│   ├── 04_explore_domainnet.ipynb
│   ├── 05_explore_core50.ipynb
│   ├── 06_measure_shift_imagenet100.ipynb
│   ├── 07_measure_shift_clear10.ipynb
│   ├── 08_measure_shift_domainnet.ipynb
│   ├── 09_measure_shift_core50.ipynb
│   └── 10_combine_shift_results.ipynb
└── part-b-modeling/
    └── (planned below — exact files to be finalized after Part A)
```

## Part A — Distribution checks (build this first, file by file)

| # | File | What it does |
|---|------|---------------|
| 1 | `01_setup_check.ipynb` | Connects to the RunPod GPU machine, confirms the GPU is visible, and checks that the libraries we need (PyTorch, Avalanche) import correctly. |
| 2 | `02_explore_imagenet100.ipynb` | Loads ImageNet-100's 20-task stream, shows a handful of sample images per task, and reports how many classes/images are in each task. |
| 3 | `03_explore_clear10.ipynb` | Same exploration as above, for CLEAR-10's 10 time buckets. |
| 4 | `04_explore_domainnet.ipynb` | Same exploration, for DomainNet's 6 style domains. |
| 5 | `05_explore_core50.ipynb` | Same exploration, for CORe50's NIC batches. |
| 6 | `06_measure_shift_imagenet100.ipynb` | Computes the MMD and CKA shift numbers between each consecutive pair of tasks in ImageNet-100. |
| 7 | `07_measure_shift_clear10.ipynb` | Same shift measurement, for CLEAR-10. |
| 8 | `08_measure_shift_domainnet.ipynb` | Same shift measurement, for DomainNet. |
| 9 | `09_measure_shift_core50.ipynb` | Same shift measurement, for CORe50. |
| 10 | `10_combine_shift_results.ipynb` | Pulls the shift numbers from files 6–9 into one table and a couple of plots, so we can see all four datasets' shift patterns side by side. **Part A is done once this file runs.** |

## Part B — Modeling (rough outline, will be refined after Part A)

| # | File (working name) | What it will do |
|---|------|---------------|
| 1 | `01_train_naive.ipynb` | Trains the no-special-trick baseline across all four datasets, scratch-trained models only, to get a forgetting baseline. |
| 2 | `02_train_regularization.ipynb` | Trains EWC and LwF (the regularization-based methods) across all four datasets. |
| 3 | `03_train_replay.ipynb` | Trains ER and DER++ (the replay-based methods) across all four datasets. |
| 4 | `04_train_pretrained_arms.ipynb` | Repeats the key runs above using pretrained models instead of scratch-trained ones, to test Question 2. |
| 5 | `05_combine_forgetting_results.ipynb` | Pulls together how much each method forgot, across all datasets and both scratch/pretrained versions. |
| 6 | `06_relate_shift_to_forgetting.ipynb` | Combines this with the Part A shift numbers and checks whether more shift lines up with more forgetting — this is where we actually answer Question 1 and Question 2. |

Part B's exact file split, and which methods/benchmarks get which, may change once we see how Part A behaves and how expensive the RunPod runs actually are — we'll revisit this list together before writing any Part B code.
