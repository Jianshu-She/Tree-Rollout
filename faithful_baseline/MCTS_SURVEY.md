# Survey: MCTS Approaches for LLM Reasoning in RL

> Comparison of tree search strategies used in LLM mathematical reasoning.
> Focus: expansion policy, branching factor, reward signal, and how tree structure relates to flat rollout distributions.

## Method Comparison Table

| Method | Paper | Venue | Expansion Strategy | Branching Factor | Reward Signal | Tree Granularity | RL Training? | Key Innovation |
|--------|-------|-------|-------------------|-----------------|---------------|-----------------|-------------|----------------|
| **Standard UCT-MCTS** | (baseline) | — | Fixed k children per expansion | k=2~8 (fixed) | Logprob / PRM | 256-token chunks | No (inference only) | Classical UCB1 selection |
| **ReST-MCTS*** | [Zhang et al. 2024](https://arxiv.org/abs/2406.03816) | NeurIPS 2024 | Expand until predefined limit; PRM-guided | Configurable limit | Process Reward Model | Step-level | Yes (self-training) | PRM guides search + iterative self-training |
| **rStar-Math** | [Microsoft, 2025](https://arxiv.org/abs/2501.04519) | Preprint | Try each action type once; up to 5 children per type | ~5 per action type | SLM-based PRM | Step-level (multiple action types) | Yes (self-evolution) | Mutual reasoning + multi-action-type expansion |
| **MCTSr** | [Zhang et al. 2024](https://arxiv.org/abs/2406.07394) | Preprint | Self-refine existing answer (not traditional expand) | 1 (refine) | Self-evaluation LLM score | Full solution | No | Self-refine as tree expansion |
| **AlphaMath** | [Chen et al. 2024](https://arxiv.org/abs/2405.03553) | Preprint | AlphaGo-style UCT + value model | Fixed k | Trained value model | Step-level | Yes | Step-level value + policy model |
| **TreeRL (EPTree)** | [THUDM, 2025](https://arxiv.org/abs/2506.11902) | ACL 2025 | Fork at top-N highest entropy tokens | Adaptive (entropy-based) | Outcome + process reward | Token-level fork points | Yes (on-policy RL) | Entropy-guided branching for RL data |
| **Entropy-Gated Branching** | [2025](https://arxiv.org/abs/2503.21961) | Preprint | Branch only at high-uncertainty steps | Adaptive (entropy threshold) | Verifier | Step-level | No (inference) | Branch only when uncertain |
| **ETTRL (ETMR)** | [2025](https://arxiv.org/abs/2508.11356) | Preprint | Fork at K highest-entropy tokens | K (top-K entropy) | Majority voting | Token-level | Yes (test-time RL) | Entropy-fork + advantage reshaping |
| **DeepSearch** | [2025](https://arxiv.org/abs/2509.25454) | Preprint | Global frontier + entropy guidance | Adaptive | Verifiable rewards | Step-level | Yes (training loop) | MCTS in RL training loop; entropy selects supervision targets |
| **AB-MCTS** | [2025](https://arxiv.org/abs/2503.04412) | Preprint | Dynamically decide wider vs deeper | Adaptive (feedback-driven) | External feedback | Full solution | No (inference) | "Go wider" or "go deeper" per node |
| **Poisson-MCTS** | (Ours, proposed) | — | Sample k from NegBin(D0) / Poisson(D1+) | Depth-adaptive, distribution-guided | TBD | 256-token chunks | TBD | Match flat rollout tree statistics |

---

## Key Dimensions of Comparison

### 1. Branching Factor Strategy

| Strategy | Methods | Pros | Cons |
|----------|---------|------|------|
| **Fixed k** | Standard MCTS, AlphaMath | Simple, predictable compute | Doesn't match natural reasoning divergence patterns |
| **Entropy-adaptive** | TreeRL, ETTRL, EGB, DeepSearch | Branches where uncertain → efficient | Requires entropy computation; may miss non-entropy-correlated branching |
| **Distribution-guided** | Poisson-MCTS (ours) | Matches observed flat rollout structure | Requires pre-computed statistics from flat rollouts |
| **Self-refine** | MCTSr | Novel: "expand" = improve existing answer | Not true tree branching; limited diversity |

### 2. Reward Signal

| Reward Type | Methods | Pros | Cons |
|-------------|---------|------|------|
| **Logprob** | Standard MCTS (ours) | No extra model needed | Weak signal; model confidence ≠ correctness |
| **Process Reward Model (PRM)** | ReST-MCTS*, rStar-Math, AlphaMath | Step-level credit assignment | Requires trained PRM; expensive |
| **Self-evaluation** | MCTSr | No external model | Unreliable for hard problems |
| **Outcome reward (verifiable)** | DeepSearch, TreeRL | Ground truth signal | Sparse; no step-level guidance |
| **Majority voting** | ETTRL | Aggregation-based | Requires multiple completions |

### 3. Integration with RL Training

| Integration | Methods | Description |
|-------------|---------|-------------|
| **Inference only** | Standard MCTS, MCTSr, EGB, AB-MCTS | Tree search at test time; no training |
| **Data collection for training** | ReST-MCTS*, rStar-Math, AlphaMath | MCTS generates training data (preferences / rewards) |
| **Embedded in training loop** | DeepSearch, TreeRL, ETTRL | MCTS runs during RL training for on-policy exploration |

### 4. What Makes Trees Different from Flat Rollouts?

| Aspect | Flat Rollouts (128 independent) | Standard MCTS (k=2) | Poisson-MCTS (proposed) |
|--------|-------------------------------|---------------------|------------------------|
| **D0 branching** | NegBin: mean=4~26, heavy-tailed | 2 (fixed) | Sample from NegBin(r,p) |
| **D3+ branching** | ~1 (fully diverged) | 2 (still splitting) | ~1 (from Poisson λ≈1) |
| **Dispersion** | 3~18× overdispersed at D0 | 0 (deterministic) | Matches observed dispersion |
| **Compute allocation** | Uniform across rollouts | UCB1-guided | Distribution-guided |
| **Diversity** | High (independent samples) | Limited (UCB1 focuses on promising paths) | Controlled (match observed diversity) |

---

## Experimental Results: Our Standard MCTS Baseline

### Configuration
- UCT with UCB1, logprob reward
- 128 MCTS iterations per problem
- 256 tokens per node, max depth 16
- k ∈ {2, 4, 8} children per expansion
- Qwen2.5-Math-7B checkpoints (step 0/40/80/120)
- 10 MATH500 problems (indices 0-9)

### Full Results: k=2, k=4, k=8 (10 problems each)

| Stage | Flat Rollout | MCTS k=2 | MCTS k=4 | MCTS k=8 |
|-------|-------------|----------|----------|----------|
| **Accuracy** | | | | |
| step_0 | 43.8% | 90% | 60% | 40% |
| step_40 | 73.3% | 80% | 80% | 100% |
| step_80 | 77.2% | 90% | 100% | 80% |
| step_120 | 79.6% | 90% | 90% | 90% |
| **Avg Nodes/Tree** | | | | |
| step_0 | — | 19.2 | 86.6 | 266.6 |
| step_40 | — | 14.4 | 61.4 | 221.8 |
| step_80 | — | 21.4 | 73.0 | 280.2 |
| step_120 | — | 11.8 | 57.4 | 272.2 |
| **D0 Avg BF** | | | | |
| step_0 | 26.0 | 2.0 | 4.0 | 8.0 |
| step_40 | 5.7 | 2.0 | 4.0 | 8.0 |
| step_80 | 4.7 | 2.0 | 4.0 | 8.0 |
| step_120 | 4.0 | 2.0 | 4.0 | 8.0 |

### Key Findings

1. **MCTS branching is always constant** (= k) at all depths — fundamentally different from flat rollout where branching decays from ~4-26 at D0 to ~1 at D3+.

2. **Larger k hurts base model**: step_0 accuracy drops from 90% (k=2) → 60% (k=4) → 40% (k=8), because the base model generates many low-quality children that dilute the search.

3. **Larger k helps trained models**: step_40 goes from 80% (k=2) → 100% (k=8), because trained models generate more consistently good children.

4. **Compute scales linearly with k**: k=8 uses ~15-20× more nodes than k=2, but accuracy improvement is marginal for trained models.

5. **No fixed k matches flat rollout**: Flat rollout D0 branching is 4-26 (stage-dependent, overdispersed), while MCTS is always exactly k. This motivates Poisson-MCTS with depth-adaptive, stochastic branching.

### DeepSearch Results (global frontier + entropy, base width=8)

| Stage | Accuracy | Avg Nodes | MaxDepth (avg) |
|-------|----------|-----------|----------------|
| step_0 | 20% | 365.4 | varies (1-16) |
| step_40 | 80% | 269.0 | 2-16 |
| step_80 | 80% | 230.5 | 2-16 |
| step_120 | 100% | 225.7 | 2-15 |

### Complete Comparison Table

| Method | step_0 | step_40 | step_80 | step_120 | Avg Nodes |
|--------|--------|---------|---------|----------|-----------|
| Flat Rollout (128) | 43.8% | 73.3% | 77.2% | 79.6% | — |
| Standard k=2 | 90% | 80% | 90% | 90% | 12-21 |
| Standard k=4 | 60% | 80% | 100% | 90% | 57-87 |
| Standard k=8 | 40% | 100% | 80% | 90% | 222-280 |
| DeepSearch | 20% | 80% | 80% | 100% | 226-365 |

### Key Findings

1. **k=2 is most compute-efficient**: High accuracy with fewest nodes (12-21). UCB1 focuses on promising paths.

2. **Larger k hurts weak models, helps strong ones**: step_0 drops from 90% (k=2) → 40% (k=8) → 20% (DeepSearch), but step_120 goes 90% → 90% → 100%.

3. **DeepSearch produces adaptive branching**: Unlike fixed-k methods, DeepSearch's trees have variable branching (width decays with depth). But with logprob reward (no trained PRM), entropy guidance can be noisy.

4. **No fixed k matches flat rollout**: Flat rollout D0 branching is 4-26, MCTS is always exactly k. This motivates Poisson-MCTS.

5. **Compute vs accuracy trade-off**: k=2 gives the best accuracy/node ratio. DeepSearch and k=8 use 10-20× more compute for marginal gains.

**Conclusion**: Standard MCTS with any fixed k does NOT match flat rollout tree structure. DeepSearch's adaptive branching is closer but still far from flat rollout distributions. This motivates distribution-guided branching (Poisson-MCTS).
