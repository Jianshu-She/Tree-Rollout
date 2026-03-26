# DeepSearch MCTS vs Flat Rollout: Structural Comparison

> Core claim: DeepSearch-style MCTS cannot replace flat rollouts.
> The tree structures are fundamentally different in branching pattern,
> diversity, and compute allocation.

## Experimental Setup

| | Flat Rollout | DeepSearch MCTS |
|---|---|---|
| **Method** | 128 independent samples, post-hoc syntactic clustering into tree | Global frontier selection + entropy guidance + dynamic width |
| **Model** | Qwen2.5-Math-7B (step 0/40/80/120) | Same |
| **Chunk size** | 256 tokens | 256 tokens |
| **Base width** | N/A (emergent) | 8 children/expansion |
| **Problems** | 400 (train set) | 10 (problems 0-9) |
| **Reward** | N/A (post-hoc correctness) | Logprob (no trained PRM) |

## Accuracy Comparison

| Stage | Flat Rollout (400p) | DeepSearch (10p) |
|-------|--------------------:|------------------:|
| step_0 (base) | 43.8% | 20% |
| step_40 | 73.3% | 80% |
| step_80 | 77.2% | 80% |
| step_120 | 79.6% | 100% |

- DeepSearch **fails on weak models** (20% vs 43.8% for base model)
- DeepSearch matches or exceeds flat rollout on trained models

## Branching Factor: The Key Structural Difference

### Flat Rollout (post-hoc tree)

| Depth | step_0 | step_40 | step_80 | step_120 |
|-------|--------|---------|---------|----------|
| D0 | **26.0** | **5.7** | **4.7** | **4.0** |
| D1 | 1.8 | 3.7 | 4.2 | 4.6 |
| D2 | 1.4 | 1.7 | 1.8 | 1.8 |
| D3 | 1.1 | 1.2 | 1.2 | 1.3 |
| D4+ | ~1.0 | ~1.0 | ~1.0 | ~1.0 |

**Pattern**: Stage-dependent, heavy-tailed at D0, rapid decay to ~1 by D3.
D0 branching is overdispersed (Negative Binomial, var/mean = 3-18x).

### DeepSearch MCTS

| Depth | step_0 | step_40 | step_80 | step_120 |
|-------|--------|---------|---------|----------|
| D0 | 8.0 | 8.0 | 8.0 | 8.0 |
| D1 | 8.0 | 8.0 | 8.0 | 8.0 |
| D2 | 7.0 | 7.0 | 7.0 | 7.0 |
| D3 | 7.0 | 7.0 | 7.0 | 7.0 |
| D4 | 6.0 | 6.0 | 6.0 | 6.0 |
| D5 | 6.0 | 6.0 | 6.0 | 6.0 |

**Pattern**: Fixed base_width=8, slow linear decay. **Same across all stages** — does not adapt to model capability.

### Divergence

| Stage | Avg relative BF divergence |
|-------|---------------------------|
| step_0 | 339% |
| step_40 | 303% |
| step_80 | 302% |
| step_120 | 300% |

DeepSearch branching is **3-4x off** from flat rollout at every stage.

## Why DeepSearch Cannot Replace Flat Rollouts

### 1. Branching is fixed, not emergent

Flat rollout branching **emerges** from the model's natural diversity:
- Base model (step_0): 26 branches at D0 — very diverse reasoning
- Trained model (step_120): 4 branches at D0 — converged to fewer strategies

DeepSearch always uses width=8 regardless of model capability. It cannot capture how RL training **reduces** branching diversity.

### 2. Wrong depth profile

Flat rollouts: most branching at D0, then **rapid collapse** to single paths by D3.
DeepSearch: wide at all depths (8→7→6), wasting compute on deep branching that flat rollouts show is unnecessary.

### 3. No variance (overdispersion)

Flat rollout D0 branching has **huge variance** (NegBin, var/mean up to 18x):
- Some problems have 1-2 branches (easy, one obvious approach)
- Some problems have 50-100 branches (hard, many diverse attempts)

DeepSearch: every problem gets exactly 8 branches. No problem-adaptive allocation.

### 4. Diversity vs exploitation trade-off

Flat rollouts: 128 **independent** samples → maximum diversity.
DeepSearch: UCT-like selection **focuses on promising paths** → reduced diversity, higher exploitation. Good for accuracy, bad for capturing the full reasoning landscape.

## Implications for Poisson-MCTS

These differences motivate our proposed Poisson-MCTS:

| Feature | DeepSearch | Poisson-MCTS (proposed) |
|---------|-----------|------------------------|
| D0 branching | Fixed 8 | Sample from NegBin(r,p) — stage-dependent |
| D1+ branching | Slow linear decay | Poisson(λ_d) — fast decay matching flat rollout |
| Problem-adaptive | No | Yes (via variance in NegBin sampling) |
| Stage-adaptive | No | Yes (different r,p per training stage) |
| Matches flat rollout | No (300% divergence) | By design (~0% divergence in distribution) |
