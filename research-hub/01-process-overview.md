# Process overview — the whole project, start to finish

> Written to be read top to bottom, by someone who hasn't looked at this in a while and doesn't remember the jargon. If a word needs explaining, it's explained the first time it shows up.

## 1. The problem we're studying

Imagine a model that learns things one topic at a time — like learning to recognize cats this week, then dogs next week, then birds the week after. A common problem is that while it's learning birds, it quietly gets worse at recognizing cats. This is called **forgetting**.

We want to know: does the *amount* a model forgets depend on how different the topics are from each other? If "cats" and "dogs" are pretty similar, maybe there's less forgetting. If "cats" and "something totally different, like X-ray images" are involved, maybe there's a lot more forgetting. We're measuring "how different two topics are" and calling that the **shift** between them.

We're also asking a second question: if the model already has some general knowledge before it starts (because it was **pretrained** — trained once on a big, unrelated dataset first, the way a student might already know general biology before studying a specific disease), does that make it *more resistant* to this kind of forgetting?

## 2. The two questions (hypotheses), in plain words

- **Question 1:** Some forgetting-prevention techniques work by replaying old examples back to the model (**replay-based** methods). Others work by penalizing the model for changing too much (**regularization-based** methods). We think the regularization-based ones will struggle more as the shift between topics gets bigger, while the replay-based ones will stay more stable.
- **Question 2:** We think a pretrained model (one that already knows general visual features before we start) will forget less than a model trained completely from scratch, especially when the shift between topics is large.

## 3. The four datasets we'll test this on

A "dataset" here just means a specific collection of images, already split up into a sequence of topics/tasks. We're using four, because we want to see if the pattern holds across very different kinds of shift:

1. **ImageNet-100** — everyday objects, split into 20 tasks. Each task introduces new object categories. Lots of tasks, good for seeing a trend.
2. **CLEAR-10** — the same categories photographed over time, so the "shift" is slow and gradual (like fashion or car styles slowly changing over the years).
3. **DomainNet** — the same categories, but drawn in very different *styles* (real photo vs. painting vs. sketch, etc.), so the shift is sudden and large.
4. **CORe50** — everyday objects seen from different angles/backgrounds, in many small steps. Lots of small tasks.

We'll look at and measure all four before doing any of the heavier modeling work.

## 4. The five techniques we'll compare

These are standard, published methods for reducing forgetting. We're not inventing new ones — we're just running existing ones and comparing them:

- **Naive** — no special trick at all. This is the baseline: how bad is forgetting with no help?
- **EWC** and **LwF** — regularization-based: they try to stop the model's important weights from changing too much.
- **ER** and **DER++** — replay-based: they keep a small memory of old examples and mix them back in while learning new ones.

## 5. How we measure "shift"

Two numbers, both comparing one task to the next:

- **MMD** — compares the *raw images* of two tasks (are the pixel statistics very different?).
- **CKA** — compares what a neural network's *internal representation* looks like for two tasks (does the model "see" them very differently, even if the raw pixels look similar?).

We compute both so we're not relying on just one way of measuring "different."

## 6. The two-part plan

**Part A — Distribution checks.** For each of the four datasets: load it, look at some example images from a few tasks so we understand what we're dealing with, then compute the MMD and CKA shift numbers between consecutive tasks. Once we have shift numbers for all four datasets, Part A is done.

**Part B — Modeling.** Only after Part A works and we've reviewed it: train the five techniques (naive, EWC, LwF, ER, DER++) on the four datasets, with and without pretraining, and measure how much each one forgets. Then check whether the forgetting lines up with the shift numbers from Part A — that's the actual test of Question 1 and Question 2 above.

See [02-file-map.md](02-file-map.md) for the specific files we'll build for each part.

## 7. How we're actually going to work

- One file at a time. I build one notebook or script, you look at it and we run it together, and we don't move to the next file until that one makes sense and works.
- Everything gets explained as we go — if a step uses a term or a technique you haven't seen, I'll explain it before we use it, not after.
- Part A (distribution checks) happens entirely before Part B (modeling) starts. We don't train any models until we've looked at and measured all four datasets.
- The actual training (Part B) runs on a rented cloud GPU machine (**RunPod**), because it needs more computing power than a laptop. Part A (just loading and measuring data) can mostly run there too, to keep things consistent.

## 8. Budget

We have about $50 total to spend on the rented GPU machine. That's enough for the plan above if we keep things lean — we're not aiming for a huge sweep of extra experiments, just enough runs to answer the two questions.

## 9. Where the old work went

Everything that existed before this reset — all the old docs and all the old code — has been moved into `phase-1-archive/`, untouched. Nothing was deleted. If we ever need to check a detail from the old, more detailed plan (for example, exact model architectures or specific past decisions), it's still there.
