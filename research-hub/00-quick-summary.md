# Quick summary

> Read this first. It's the whole project in one minute. For the full walkthrough, read [01-process-overview.md](01-process-overview.md). For the list of files we're going to build, read [02-file-map.md](02-file-map.md).

**The question:** when a computer-vision model learns a sequence of tasks one after another, does it forget old tasks *more* when the tasks look very different from each other (a big "shift")? And does starting from a model that was pretrained on another dataset make it forget less?

**The plan, in two parts:**

1. **Distribution checks** — look at four different datasets, one at a time, and measure how much each one "shifts" from task to task.
2. **Modeling** — train several standard forgetting-prevention techniques on those same datasets and see whether the amount of shift predicts how much forgetting happens.

**How we're working now:** one file at a time. I write one notebook or script, you look at it, we run it, and only then do we move to the next one. No big batches of code dropped at once.

**Where things live:**
- `research-hub/` — all the plain-language docs, plans, and notes (this folder).
- `cl-nonstationarity-research/` — the actual code, started fresh.
- `phase-1-archive/` — everything from before this reset (old docs + old code), kept for reference, untouched.
