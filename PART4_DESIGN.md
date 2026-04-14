# Part 4: End-to-End RL Training Experimental Design

**Status**: Design phase (2026-04-14)
**Priority**: HIGH — paper necessity (see PROGRESS_REPORT §4.5)
**Goal**: Prove "drift is harmful" with gold-standard evidence — train 4 policies with different rollout methods, evaluate each on held-out benchmarks, show drift-prone methods produce worse final models.

---

## 1. Why Part 4 Exists

Our current offline claims characterize drift *descriptively* — DS has lower Pearson correlation with Flat's per-problem accuracy, higher no-advantage rate, higher token cost. But we **cannot prove drift is harmful** without running actual RL training. This is a critical gap that Qirong flagged at the 2026-04-09 meeting: **"MUST run actual RL training"**.

## 2. Core Experimental Matrix

| Experiment | Rollout method | Purpose |
|---|---|---|
| **E1 (baseline)** | Flat (standard GRPO) | Reproduce step_0 → step_120 trajectory |
| **E2 (faithful-BFS)** | BFS Tree | Show faithful tree method matches or beats flat |
| **E3 (faithful-NegBin)** | NegBin MCTS | Show faithful tree method matches or beats flat |
| **E4 (drifted-DS)** | DeepSearch | Show drift-prone method produces worse final policy |

**Key design decision**: ALL 4 experiments use **identical everything except rollout method**. Same data, same optimizer, same hyperparameters, same total training compute, same evaluation.

## 3. Training Configuration

### Shared settings (same for all 4 experiments)
- **Base model**: Qwen2.5-Math-7B
- **Algorithm**: GRPO (`adv_estimator=grpo`), no KL in reward, no KL loss
- **Optimizer**: AdamW, lr=1e-6, cosine schedule
- **Clip ratio**: [0.2, 0.28]
- **Max prompt length**: 2K; Max response length: 8K
- **Train data**: MATH500 train split (400 problems; need to generate parquet from `data-prepare/data/MATH500_train.json`)
- **Eval data**: MATH500 test (500 problems) + AIME-2024 (30 problems)
- **Total steps**: 200 (short run) → 1000 (full run if short looks good)
- **Batch size**: 512 prompts (TBD based on GPU memory with tree rollout overhead)
- **Reward function**: `compute_score_boxed` (answer extraction + exact match) from verl/utils/reward_score

### Rollout-specific settings

| Setting | Flat | BFS | NegBin | DeepSearch |
|---|---|---|---|---|
| n per prompt | 32 | natural (~10-15) | natural (~10-15) | 8 fixed |
| max_depth | — | 16 | 16 | 64 |
| branching_factors | — | stage-0 fitted (TODO: online refit) | stage-0 fitted | 8 fixed |
| UCB1 (alpha, C) | — | — | (0.5, 1.414) from step_0 BO | — |
| DS global weights | — | — | — | (0.4, 0.4, 0.01) from paper |

**Note on n**: Under the "same compute" framing, n varies per method. Flat gets 32, tree methods get whatever compute budget they use naturally. This needs to be aligned to per-prompt compute, not per-prompt trajectories.

## 4. Compute Budget

Based on offline data:
- Flat at n=32 per prompt ≈ 52K tokens/prompt
- BFS/NegBin ≈ 22-25K tokens/prompt (plateau)
- DS ≈ 64K tokens/prompt (higher than flat at n=32)

For "same wall-clock GPU time" comparison, we should **not** match per-step compute — we should run each method for the same number of training steps. Tree methods will use less compute per step, so they'll finish faster. This is the natural deployment scenario.

Alternative: match per-prompt compute by adjusting n for flat down to match tree methods. Pick ONE of these and document it.

**Recommendation**: Match per-step (same #prompts × n) and let tree methods finish faster. Under this setup, tree methods' advantage includes both "better per-step" AND "more steps per GPU hour".

## 5. Evaluation Protocol

### What we evaluate
All 4 trained models evaluated identically:
- MATH500 test: pass@1, pass@32, majority@32 (same as training benchmark)
- AIME-2024: pass@1, pass@16 (harder, out-of-distribution)
- GSM8K (optional): pass@1 (easier, sanity check on forgetting)

### How we evaluate
**All evaluation uses flat/standard sampling** (temperature=0.7, top_p=0.95, n=32). No tree methods at evaluation time. This is the critical fairness point — we're measuring which training method produces the best model under **standard deployment conditions**.

### Per-step metrics to log (during training)
- Mean train reward
- Reward distribution: std, bimodality coefficient, % no-advantage prompts
- KL(π_t || π_base) — drift from base model
- Eval accuracy on MATH500 test (every 20 steps)
- Eval accuracy on AIME-2024 (every 50 steps)
- Gradient norm, learning rate
- Tokens consumed per step (for compute accounting)

## 6. Predicted Outcomes (pre-registration)

| Outcome | Flat final acc | BFS/NegBin final acc | DS final acc | Paper claim |
|---|---|---|---|---|
| **A: BFS/NegBin win, DS loses** | baseline | +3 to +6 pp | −2 to −5 pp | Main thesis confirmed |
| **B: BFS/NegBin match, DS loses** | baseline | ±1 pp | −2 to −5 pp | Weaker but still valid |
| **C: BFS/NegBin win, DS matches** | baseline | +3 to +6 pp | ±1 pp | Half-win |
| **D: All methods converge** | baseline | ±1 pp | ±1 pp | Saturation — needs harder eval |
| **E: DS wins** | baseline | ±1 pp | +3 pp | Paper fails |

My estimated probabilities: A=50%, B=20%, C=15%, D=10%, E=5%.

## 7. Infrastructure Plan

### Repo layout (proposed)
```
MCTS/
├── verl_tree_rl/                   # New subdir, does NOT touch copus/verl
│   ├── README.md
│   ├── tree_rollout.py             # FaithfulRollout wrapper
│   ├── tree_engines/
│   │   ├── base.py                 # Abstract engine interface
│   │   ├── flat_engine.py          # pass-through wrapper
│   │   ├── bfs_engine.py           # port from mcts_inference/bfs_engine.py
│   │   ├── negbin_engine.py        # port from mcts_inference/poisson_mcts_engine.py
│   │   └── deepsearch_engine.py    # port from mcts_inference/deepsearch_engine.py
│   ├── register.py                 # Hook FaithfulRollout into verl's rollout registry
│   ├── recipes/
│   │   ├── run_grpo_flat.sh
│   │   ├── run_grpo_bfs.sh
│   │   ├── run_grpo_negbin.sh
│   │   └── run_grpo_deepsearch.sh
│   ├── config/
│   │   └── tree_grpo.yaml          # extends verl default, adds rollout.method field
│   └── metrics/
│       └── drift_logger.py         # log per-step drift metrics during training
├── PART4_DESIGN.md                 # this file
└── ...
```

### Integration strategy
1. Do NOT modify `/mnt/weka/home/jianshu.she/copus/verl/` — it's the shared tooling
2. The new `verl_tree_rl/tree_rollout.py::FaithfulRollout` class **wraps** verl's `VLLMRollout` (or `SGLangRollout`). Instead of calling vLLM once per prompt, it runs a tree engine that calls vLLM multiple times per prompt (once per expansion level)
3. `register.py` monkey-patches `verl.workers.fsdp_workers.get_rollout_class` to recognize `rollout.name="tree_faithful"`, with the tree method specified by `rollout.tree_method` (flat/bfs/negbin/deepsearch)
4. Training launches via `PYTHONPATH=MCTS/verl_tree_rl:$PYTHONPATH python -m verl.trainer.main_ppo --config-path MCTS/verl_tree_rl/config ...`

### Why wrap instead of replace
- We reuse all of verl's DataProto handling, FSDP wiring, reward evaluation, GRPO advantage computation, checkpointing, logging
- Our tree engines only need to output a DataProto compatible with what verl's vLLM rollout produces — exact same fields, exact same shapes
- DeepSearch's official repo (`deepsearch/rollout/sglang_mcts_wrapper.py`) uses this same wrapping strategy — we can crib their DataProto-packing logic

### Risks
1. **DataProto packing correctness**: tree methods produce variable n per prompt, but verl expects fixed-shape batches. Solution: pad to fixed n, mask padded positions
2. **vLLM re-initialization overhead**: tree engine makes many sequential vLLM calls. verl's `generate_sequences` wraps `LLM.generate`; we need to make sure vLLM isn't re-initialized each call. It shouldn't be — verl holds a persistent vLLM handle
3. **Advantage computation**: GRPO advantages are computed per-prompt group. If tree method produces variable n per prompt, we need per-prompt normalization. verl already does this — it should just work
4. **FSDP / TP interaction**: if trainer runs with TP=2, each rollout worker holds half the model. Tree engine must call vLLM through the rollout worker, not spawn its own. Verified: DeepSearch's wrapper does this correctly by holding a reference to the rollout class instead of instantiating its own

## 8. Phased Execution

### Phase 1: Scaffold (2-3 days, no GPU)
- [ ] Set up `verl_tree_rl/` directory structure
- [ ] Write `FaithfulRollout` wrapper with only `flat` pass-through (other methods return NotImplementedError)
- [ ] Write `register.py` hook
- [ ] Write `run_grpo_flat.sh` recipe
- [ ] **Milestone**: flat pass-through should produce identical DataProto to native verl vLLM rollout

### Phase 2: Flat smoke test (1 day, few GPU-hours)
- [ ] Run E1 (Flat) for 10 steps on 2 problems to validate the pipeline doesn't crash
- [ ] Verify training logs make sense (reward, adv_std, KL, etc.)
- [ ] **Milestone**: can reproduce one step of the existing copus/verl baseline exactly

### Phase 3: Port tree engines (3-5 days, incremental GPU)
- [ ] Port `bfs_engine.py` → `tree_engines/bfs_engine.py` (handle DataProto packing)
- [ ] E2 (BFS) smoke test on 10 problems × 10 steps
- [ ] Port `poisson_mcts_engine.py` → `tree_engines/negbin_engine.py`
- [ ] E3 (NegBin) smoke test on 10 problems × 10 steps
- [ ] Port `deepsearch_engine.py` → `tree_engines/deepsearch_engine.py`
- [ ] E4 (DeepSearch) smoke test on 10 problems × 10 steps
- [ ] **Milestone**: all 4 methods can run 10 training steps without crash

### Phase 4: Short runs (1 week, sustained GPU)
- [ ] Run E1-E4 each for 200 steps, log all drift metrics
- [ ] Look for early signs of matching the predicted outcomes (A-E)
- [ ] Decision point: proceed to full run, or debug

### Phase 5: Full runs (2 weeks, sustained GPU)
- [ ] Run E1-E4 each for 1000 steps
- [ ] Evaluate at step 0, 100, 200, 500, 1000 on MATH500 test + AIME
- [ ] Generate paper figures: reward dist evolution, KL drift, eval accuracy curves
- [ ] **Milestone**: paper's Part 4 results are done

## 9. Open Design Questions

1. **Train data**: MATH500 train (400 problems) or dapo-math-17k (17K problems)?
   - MATH500 = consistent with offline analysis but small
   - dapo-math-17k = consistent with step_0/40/80/120 checkpoints but different from offline
   - **Decision TBD**: lean MATH500 if we want self-consistency, dapo if we want to extend existing ckpts

2. **Matching compute vs matching steps**: all methods run for same #steps (recommendation) or same total compute?

3. **Do we also test fixed (alpha, C) vs per-stage BO (Item #9B)?** This doubles the experiment count (8 runs instead of 4). Only needed if the fixed-param versions fail or look borderline.

4. **Seed variance**: how many random seeds per method? 1 (fast) or 3 (statistical significance)?
   - Recommendation: 1 for short runs, 3 for the final full runs of the best methods

5. **Eval during training**: every N steps. N=20 is standard but expensive (full MATH500 eval ≈ 30 minutes). Can we reduce by sampling?

## 10. Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-14 | Part 4 upgraded from LOW to HIGH priority | Offline drift claims are descriptive only; gold-standard proof requires actual RL training |
| 2026-04-14 | Wrap verl's rollout, don't fork | Reuse verl's infrastructure; DeepSearch's repo validates this pattern |
| 2026-04-14 | 4 methods × 1 seed × 200→1000 steps | Feasible within 2-3 weeks GPU budget |
