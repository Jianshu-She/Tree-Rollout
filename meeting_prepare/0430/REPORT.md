# Tree-RL Project Progress Report
**Date**: 2026-04-30
**Author**: Jianshu She
**Target venues**: EMNLP 2026 / ICLR 2027

---

## 1. Executive Summary (one paragraph)

We are building a paper that asks: **what happens to the natural rollout distribution when you replace flat sampling with tree search in RL post-training?** Existing MCTS-RL papers cherry-pick favorable metrics. Our approach has three layers:

1. **Empirically characterize** flat rollout's natural tree structure by post-hoc clustering 128-rollout outputs into syntactic trees and fitting branching-factor / accuracy distributions per depth.
2. **Propose "faithful tree methods"** (BFS Tree + NegBin MCTS) that explicitly mimic the fitted distribution to harvest tree-search compute savings without changing the model's natural distribution.
3. **Validate end-to-end with GRPO RL training** on Qwen2.5-Math-7B + DAPO-MATH, comparing 4 rollout methods (Flat / BFS / NegBin / DeepSearch) on identical training/eval setups.

**Current status (2026-04-30)**: Parts 1-2 (offline analysis) complete, Part 3 (end-to-end RL) results in: BFS 36.4% > NegBin 26.9% > Flat 23.45% on MATH-500-test after 100 GRPO steps. DeepSearch (literature baseline) is currently running.

---

## 2. Part 1 — Empirical Characterization

### 2.1 Post-hoc clustering of flat rollouts into trees

For each problem in MATH500 train (400 problems), we run 128 flat rollouts under Qwen2.5-Math-7B at 4 RL training stages (step_0 / step_40 / step_80 / step_120). The 128 trajectories are post-hoc clustered into a tree by aligning prefixes:

- **Syntactic clustering** (n-gram / token-level prefix match): used for our main analysis
- **Semantic clustering** (GPT-4o-mini judge): used as a cross-validation in the offline section

Each problem yields a tree where common prefixes are merged into shared nodes; divergent suffixes become separate branches.

**Outputs**:
- `faithful_baseline/results/math500_full/train/trees_syntactic/step_{0,40,80,120}/tree_step_X_pY.json` — 400 trees per stage
- `faithful_baseline/results/math500_full/train/rollouts_step_X.json` — raw flat rollout data (~38 GB total)

### 2.2 Distribution fitting at each tree depth

For each (stage, depth), aggregate branching factors across all 400 trees and fit candidate distributions:

| Depth | Best fit (AIC winner) | Reasoning |
|---|---|---|
| **D0** (root → first chunk) | **Negative Binomial** (heavy-tailed, r/p) | Variance >> mean → NegBin needed |
| **D1** | **Geometric** (single-parameter heavy tail) | Surprising finding: beats NegBin despite fewer params |
| **D2+** | **Poisson** (mean ≈ variance) | Tail decays, Poisson sufficient |

For step_0 specifically: bf[0]=26 (NegBin mean), bf[1]=1.85 (Poisson λ), bf[2]=1.37, bf[3]=1.14, bf[4]=1.06, bf[5+]≈1.

Node-accuracy distributions are fit with Beta (α, β).

**Outputs**:
- `faithful_baseline/results/math500_full/train/poisson_beta_analysis/fitted_parameters.json` — full fit
- `faithful_baseline/results/math500_full/train/poisson_beta_analysis/bf_distribution_comparison.json` — AIC scores
- `faithful_baseline/compare_bf_distributions.py` — fitting + AIC code
- `figures/distribution_fitting/bf_distribution_comparison.png` — visual

### 2.3 WBSP curves (Width / Branching / Surviving / Purity)

Per-depth aggregate metrics across 400 trees, plotted with ±1 std bands and bucketed by problem difficulty (Easy ≥0.8 acc / Medium 0.5-0.8 / Hard <0.5).

**Key findings**:
- **D0 std band is much wider than mean** → visually confirms NegBin heavy tail
- **Hard problems retain larger Width and lower Purity through step_120** → tree search has more value on hard problems
- Width / Branching collapse to ~1 by depth 5-6 → **multi-step search is only valuable at shallow depths**

**Outputs**:
- `faithful_baseline/analyze_branching_factors.py`, `plot_wbsp_by_difficulty.py`
- `figures/training_evolution/tree_curves_WBSP.png`, `tree_curves_WBSP_by_difficulty.png`

### 2.4 100-problem 4-method offline comparison @ step_0

Using the fitted parameters, we built 4 rollout methods (Flat / BFS / NegBin / DeepSearch) and compared on 100 MATH500 problems at step_0:

| Method | Accuracy | Compute (% Flat) | All-correct ✓ | All-wrong ✗ | Pearson vs Flat |
|---|---|---|---|---|---|
| Flat | 45.6% | 100% (reference) | 0 | 5 | 1.000 |
| **BFS** | **50.8% (+5.2)** | **11.3%** | 8 | 9 | **0.947** |
| **NegBin** | **51.7% (+6.1)** | **10.1%** | 9 | 15 | **0.933** |
| **DeepSearch** | 44.3% (-1.3) | 29.4% | 14 | 17 | **0.759** |

Faithful methods dominate Flat in both axes (better + cheaper). DeepSearch is dominated.

**Outputs**:
- `poisson_mcts/results/advantage_comparison/step_0/comparison_step_0.json` (869 KB)
- `poisson_mcts/results/advantage_comparison/step_0/{summary,drift_correlation_matrix,pareto_accuracy_vs_tokens,...}.png` — 10 plots
- `poisson_mcts/results/advantage_comparison/step_0/README.md` — figure-by-figure description

---

## 3. Part 2 — Faithful Tree Methods

### 3.1 BFS Tree (Method A — deterministic)

- Layer-by-layer expansion using **mean** of fitted distribution at each depth
- bf[d] = round(fitted poisson_lambda or NegBin mean)
- No UCB1, no selection bias — pure structural reproduction of flat
- **Goal**: "tree introduces no side effects" — match flat's distribution exactly
- step_0 fitted bfs: [26, 2, 1, 1, 1, ...]
- Implementation: `verl_tree_rl/tree_engines/bfs_engine.py`

### 3.2 NegBin MCTS (Method B — distribution-guided MCTS)

- Standard MCTS loop (select → expand → backpropagate)
- Branching factor at each layer is **sampled** from fitted distribution (NegBin at D0, Poisson at D1+)
- α parameter (default 0.5) controls UCB1 exploration vs random selection
- target_terminals=32, max_rollouts=64
- **Goal**: improve accuracy via UCB1 selection while remaining structurally faithful
- Implementation: `verl_tree_rl/tree_engines/negbin_engine.py`

### 3.3 DeepSearch (literature baseline — arxiv 2509.25454)

- **Global frontier selection** (across the whole tree, not depth-by-depth)
- Selection score F(s) = 0.4·tanh(Q_parent) + 0.4·entropy + 0.01·sqrt(d/d_T)
- **Token-level entropy** as the drift signal
- Fixed expansion_width=8 children per expand
- max_depth=64 (paper default), max_rollouts=64
- Hyperparameters strictly aligned with official repo (github.com/smiles724/DeepSearch)
- Implementation: `verl_tree_rl/tree_engines/deepsearch_engine.py`

### 3.4 Common engine infrastructure

All three tree engines are built on top of an inner verl rollout (vLLM) and use **batched forest expansion**:
- Each MCTS/BFS iteration selects nodes across all trees in the batch and runs a single batched `generate_sequences` call
- Without batching: 256 trees × ~10 sequential gen calls each = unfeasibly slow (>20 hours per training step)
- With batching: ~3-65 batched gen calls total per training step (depending on tree depth) = ~80-360s per step

`verl_tree_rl/tree_rollout.py` is a thin wrapper that:
1. Routes `generate_sequences` to the chosen tree engine when training
2. Falls back to flat pass-through during validation (so eval is fast and faithful to standard inference)

---

## 4. Part 3 — End-to-End RL Training (THIS REPORT'S FOCUS)

### 4.1 Setup

| Config | Value | Notes |
|---|---|---|
| **Base model** | Qwen2.5-Math-7B | step_0 checkpoint |
| **Algorithm** | GRPO (verl) | adv_estimator=grpo, no KL in reward, no KL loss |
| **Train data** | `dapo-math-15k-train-clean.parquet` | 15k unique prompts, **dedup'd** to remove 495/500 overlap with eval |
| **Eval data** | `dapo-math-500-test.parquet` | 500 MATH-domain problems (subset of `dapo-math-2k-test`) |
| **Reward** | DAPO `compute_score_boxed` | rule-based answer extraction + exact match, ±1 |
| **Total steps** | 100 | short paper run; full run target = 1000 |
| **Train batch** | 32 prompts × n=8 = 256 trajectories per step | |
| **Mini-batch** | 8 | |
| **Learning rate** | 1e-6, 10-step warmup | |
| **Test freq** | every 10 steps | 10 val points per learning curve |
| **Val mode** | flat sampling (vLLM) | tree methods only used in training |
| **Val n** | 4 samples per eval prompt | 500×4 = 2000 rollouts/val |
| **GPUs** | 8 × H100 | TP=2, FSDP=8 |
| **max_actor_ckpt_to_keep** | 1 | only latest ckpt retained (each ~86GB) |

### 4.2 Tree-method-specific configs

| Method | max_depth | tokens_per_step | bf source | target_terminals |
|---|---|---|---|---|
| Flat | — | — | — | n=8 (verl) |
| BFS | 3 | 512 | fitted step_0 [26, 2, 1, ...] | natural (~32) |
| NegBin | 12 | 512 | fitted step_0 (sampled) | 32 (capped by max_rollouts=64) |
| DeepSearch | **64** (paper) | 512 | expansion_width=8 (fixed) | 32 (capped by max_rollouts=64) |

### 4.3 Results (val accuracy on MATH-500-test)

**Methods completed**:

| Step | Flat | BFS (fitted) | NegBin |
|---|---|---|---|
| 10 | 0.5% | 0.3% | 0.55% |
| 20 | 2.05% | 3.8% | 4.45% |
| 30 | 7.95% | 12.5% | 14.05% |
| 40 | 14.2% | 16.15% | 17.0% |
| 50 | 17.7% | 24.4% | 20.55% |
| 60 | 19.8% | 31.0% | 21.0% |
| 70 | 18.4% | 33.1% | 19.9% |
| 80 | 21.5% | 36.8% | 21.35% |
| 90 | 22.95% | 36.45% | 19.75% |
| **100** | **23.45%** | **36.4%** | **26.9%** |

**Final ranking**: BFS (36.4%) > NegBin (26.9%) > Flat (23.45%).

**DeepSearch**: currently running (started 2026-04-30 03:54), expected ~6-8 hours.

### 4.4 Per-step compute

| Method | seconds/step (avg) | Reason |
|---|---|---|
| Flat | 55-65 | one vLLM call per prompt |
| BFS (fitted, max_depth=3) | 80-240 | 3 batched depth-level expansions, bf[0]=26 → 3x bf[0]=8 cost |
| NegBin (max_rollouts=64) | 330-390 | 64 MCTS iterations, each one batched gen call |
| DeepSearch (max_depth=64) | TBD | reporting after run |

### 4.5 Training/Eval data dedup (critical fix)

We discovered partway through that 495 of the 500 eval prompts in `dapo-math-2k-test`'s first-500 subset were duplicated in the original `dapo-math-17k` training set. Initial Flat run on the unfiltered data was effectively memorizing the eval. We:
1. Identified the overlap (`grep` prompt-text).
2. Removed all 495 duplicates from `dapo-math-17k`, sampled 15k unique prompts from the remainder.
3. Saved as `dapo-math-15k-train-clean.parquet` (1.54M rows, 15k unique).
4. **Re-ran all methods** on the clean data — these are the numbers above.

---

## 5. Infrastructure (engineering details)

### 5.1 verl integration

We did **not fork verl**. Instead, we registered a new rollout class `tree_faithful` into verl's `_ROLLOUT_REGISTRY` (one-line addition to `copus/verl/verl/workers/rollout/base.py`). All other code is in our `MCTS/verl_tree_rl/` module (Python path injected at launch time).

The tree rollout is a **wrapper** around verl's vLLM rollout:
1. verl trainer calls `TreeFaithfulRollout.generate_sequences(prompts)`
2. Wrapper inspects `prompts.meta_info["validate"]` — if True, falls through to inner vLLM (flat eval); if False, dispatches to the configured tree engine
3. Tree engine builds forest with batched gen calls to the inner vLLM, returns a DataProto with `[bsz × n, prompt_len + response_len]` shape matching what the inner vLLM would produce

### 5.2 Critical bugs found and fixed during development

| Bug | Symptom | Fix |
|---|---|---|
| EOS detection used `attention_mask[-1]` | Tree always terminated at depth 1 | Use `valid_tokens < tokens_per_step` instead — verl pads to full response_length |
| Default `[8,2,1]` branching factor (no fitted_params_path wired) | BFS was not actually faithful | Wire `fitted_params_path` and `training_stage` through engine_kwargs |
| Validation ran tree search | val time exploded to 30+ min/round | Skip tree engine when `meta_info["validate"]==True` |
| Train/eval overlap (495/500) | Eval was effectively training | Build clean 15k train parquet without test-prompt overlap |
| Logs in claude task tmp dir | Lost when claude session refreshed | tee stdout to `verl_tree_rl/results/<method>/log_<timestamp>.txt` |
| Per-prompt serial tree building | 26+ hours per training step | Batched forest builder: collect frontier across trees, single batched gen call |

### 5.3 Repo layout

```
MCTS/
├── ROADMAP.md                                   # long-term plan
├── PROGRESS_REPORT.md                           # mid-project status
├── PART4_DESIGN.md                              # RL training design doc
├── meeting_prepare/
│   ├── 0423/README.md
│   └── 0430/REPORT.md                           # this file
├── faithful_baseline/                           # Part 1 offline analysis
│   ├── analyze_branching_factors.py
│   ├── compare_bf_distributions.py
│   ├── plot_wbsp_by_difficulty.py
│   └── results/math500_full/train/
├── poisson_mcts/                                # Part 1-3 offline comparison
│   ├── compare_advantages.py
│   ├── plot_advantage_comparison.py
│   └── results/advantage_comparison/step_0/
├── mcts_inference/                              # offline tree engines
│   ├── bfs_engine.py / bfs_tree.py
│   ├── poisson_mcts_engine.py / poisson_mcts_tree.py
│   └── deepsearch_engine.py / deepsearch_tree.py
├── verl_tree_rl/                                # Part 4 RL infrastructure
│   ├── tree_rollout.py                          # FaithfulRollout wrapper
│   ├── tree_engines/
│   │   ├── bfs_engine.py
│   │   ├── negbin_engine.py
│   │   └── deepsearch_engine.py
│   ├── recipes/run_grpo_flat.sh                 # launch script
│   ├── results/<method>/log_*.txt               # persistent logs
│   └── analysis/parse_run_log.py                # learning curve plotter
└── figures/                                     # paper figure dir
    ├── distribution_fitting/
    ├── training_evolution/
    └── branching_analysis/
```

---

## 6. What's Done vs. What's Pending

### ✅ Done
- Part 1: empirical characterization (WBSP, distribution fits, AIC comparison)
- Part 2: faithful method designs + offline comparison @ step_0
- Part 3 — engineering: all 4 rollout methods plumbed into verl GRPO with batched forest expansion
- Part 3 — runs: Flat, BFS, NegBin (100 GRPO steps each, full learning curves)
- Test-set dedup, persistent logging, validation-skips-tree fix

### 🟡 In progress
- Part 3 run — **DeepSearch** (started 2026-04-30 03:54)

### ☐ Pending
- Part 3 — repeat all runs at later policy stages (step_40 / step_80 / step_120 base ckpts) to test "drift evolves with training"
- Part 3 — full 1000-step runs for each method
- Part 4 — Cluster Entropy MCTS or PRM-based MCTS as additional baselines
- Part 5 — Crown-jewel BO/non-myopic MCTS
- Paper write-up

---

## 7. Open Questions for Discussion

1. **Trajectory count and group composition** — currently each tree produces ~13-43 terminals but verl's pre-expansion (bsz × n=8) makes my engine output exactly 1 trajectory per expanded prompt. We discussed this is "not strictly faithful" since natural tree statistics (shared-prefix correlation) are diluted. Worth re-engineering for full faithfulness, or stick with current setup for paper time?

2. **Why does BFS dominate NegBin in RL?** Offline at step_0, NegBin slightly edged BFS (51.7% vs 50.8%). In RL, BFS leads by ~10pp. Hypothesis: UCB1 in NegBin over-exploits one good path, reducing diversity at later steps. Worth investigating α sweep?

3. **How "fair" is the comparison?** Per-step BFS uses ~3x compute of Flat; NegBin ~6x. If we equalize wall-clock or total GPU-hours, ranking might change. Should we add a compute-matched experiment?

4. **DeepSearch is the literature baseline** — paper hyperparameters strictly applied. If DS underperforms here, is that a "DeepSearch is bad" finding (compelling for paper) or "we mis-implemented" (need to debug)?

5. **Eval set size** — currently 500 problems × 4 samples = 2000 rollouts per validation. Is this enough for reliable accuracy estimates? Variance at the 21% level seems ~±2pp.

---

## 8. Suggested Meeting Agenda

1. Walk through Section 4.3 results (5 min)
2. Discuss open question #1 (trajectory count) — decide whether to re-engineer (15 min)
3. Discuss DeepSearch result once it lands (10 min)
4. Plan next steps: 1000-step runs vs. additional baselines vs. crown jewel (15 min)
5. Paper outline / figure dump for §3-4 (15 min)
