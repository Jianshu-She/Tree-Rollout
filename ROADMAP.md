# Tree-RL Project Roadmap

**Last updated**: 2026-04-11
**Target venues**: EMNLP (deadline ~end of May 2026) or ICLR 2027 (September 2026)
**NOT targeting**: NeurIPS 2026 (5/4 too tight)

## Paper Narrative (from 2026-04-09 meeting with Qirong & Raul)

**Core thesis**: Naively replacing flat rollout with tree-based MCTS in RL post-training has hidden costs. Most MCTS-RL papers cherry-pick results and don't characterize the trade-offs. We:
1. **Empirically characterize** what flat rollouts actually look like as trees (the first half of the paper).
2. **Propose "faithful tree" methods** (BFS Tree + NegBin MCTS) that mimic flat rollout structure for token efficiency without unwanted side effects.
3. **Compare against existing MCTS baselines** to show what goes wrong when you don't preserve the natural rollout distribution.
4. **Crown jewel**: A Bayesian Optimization / non-myopic search MCTS that's more aggressive but still avoids the failure modes.

**Naming convention** (locked in at meeting):
- `Poisson MCTS` → `NegBin MCTS` (because D0 is fitted with negative binomial, not Poisson)
- `BFS Tree` and `NegBin MCTS` are collectively the **"faithful methods"**
- `pure problems` → `no-advantage problems` (and split into all-correct vs all-wrong)

---

## Status Legend
- ☐ Not started
- 🟡 In progress
- ✅ Done
- ❌ Blocked / dropped

---

## Part 1: Empirical Characterization of Flat Rollout Tree Structure

### 1.1 Distribution Fitting
- ✅ Fit branching factors at each depth across 4 RL stages (step_0/40/80/120)
- ✅ Identify NegBin as better fit than Poisson for D0 (heavy-tailed)
- ✅ Identify Poisson as fit for D1+
- ✅ Beta distribution for node accuracy (purity)
- ☐ **Add a third candidate distribution** for branching (Qirong's request: don't claim NegBin is "the" distribution, compare 3)
- ☐ **Add 1-2 alternative distributions** for accuracy (compare against Beta)
- ☐ **Explore non-parametric option** (Raul's suggestion — more flexible, less assumption)
- ☐ Statistical comparison report (KS test / log-likelihood / AIC across candidate distributions)

### 1.2 Tree Structure Plots — Need Improvements
- ✅ Width / Branching / Survival / Path-length (WBSP) curves
- 🟡 **Add std bars / shaded confidence regions** to all WBSP plots (Qirong asked for "violin-like" visualization)
- ☐ **Split WBSP plots by problem difficulty** (easy vs medium vs hard)
  - "Easy" = problems converging to reward ~1 by step_120
  - "Hard" = problems still <0.5 reward by step_120
  - Show how branching/survival differ between buckets
- ☐ Verify the observation: width decreasing because rollouts terminate early (this is now explicit and important to call out)

### 1.3 Key Empirical Finding (already established)
- ✅ Variety of inference is concentrated in first few depths
- ✅ Beyond depth 5-6, branching is essentially worthless (priority/accuracy locks in)
- → **Implication**: Tree algorithms should focus search effort on shallow depths

---

## Part 2: Faithful Tree Methods (Our Methods)

### 2.1 BFS Tree
- ✅ Implementation (`mcts_inference/bfs_tree.py`, `mcts_inference/bfs_engine.py`)
- ✅ Uses fitted mean BF per depth, deterministic
- ⚠️ **Issue**: Average ~43 trajectories per problem (target 128). BF product insufficient.
- ☐ Decide: pad to 128 more aggressively, or report fewer trajectories as a feature

### 2.2 NegBin MCTS (formerly Poisson MCTS)
- ✅ Implementation (`mcts_inference/poisson_mcts_engine.py`, `poisson_mcts_tree.py`)
- ✅ Uses NegBin at D0, Poisson at D1+, with UCB1 selection (alpha-controlled)
- ⚠️ **Issue**: Average ~47 trajectories, range 1-128 (highly variable)
- ☐ **Rename across codebase**: `poisson_mcts` → `negbin_mcts`
- ☐ **Update all plots / reports / file names** to reflect new name

### 2.3 100-Problem Comparison (Done)
- ✅ Run on step_0, 100 problems
- ✅ Generate 8+ comparison figures
- ✅ Per-problem accuracy comparison (4×25 horizontal bars)
- ✅ Mean/std accuracy comparison
- ☐ **CRITICAL: Re-do "no-advantage" analysis splitting into:**
  - `all_correct` (all reward = 1) — actually a **good** sign (model solved it)
  - `all_wrong` (all reward = 0) — the **bad** case
  - Current report combines them, which is misleading
- ☐ Also report per-method: how many problems became `all_correct` only because of tree (i.e. tree solved a problem flat couldn't)

---

## Part 3: Baselines (Other MCTS Methods)

Need to position our faithful methods against existing literature.

### 3.1 Cluster Entropy MCTS — **REQUIRED**
- ☐ Implement cluster-entropy-based branching policy
- ☐ Run on same 100 problems for direct comparison

### 3.2 PRM-based MCTS — **MAYBE**
- ☐ Investigate: how widely is PRM-based MCTS actually used? (Ask IFM colleagues)
- ☐ If common enough, implement and compare
- ☐ If not, drop and explain in related work

### 3.3 Method Comparison Matrix
Once all 4 methods (Flat, BFS, NegBin MCTS, ClusterEntropy [+ PRM?]) are run:
- ☐ Side-by-side accuracy / token / no-advantage / advantage variance
- ☐ Plot: which methods are "faithful" (close to flat distribution) vs "drift" methods
- ☐ Ablation: identify which method properties cause which failure modes

---

## Part 4: End-to-End RL Training (The Key Validation)

This is the **deciding experiment** — Qirong stressed that we MUST run actual RL training to show real consequences.

### 4.1 Setup
- ☐ Pick model: Qwen2.5-Math-7B (already have)
- ☐ Pick dataset: MATH500 train (full 400 problems eventually)
- ☐ Use verl (or similar) for GRPO training loop
- ☐ Define training stages: short run (e.g. 200 steps) for ablation, long run for final results

### 4.2 Experiments
- ☐ **Baseline 1**: Standard flat-rollout GRPO (reproduce step_0 → step_120 we already have)
- ☐ **Experiment A**: BFS Tree GRPO
- ☐ **Experiment B**: NegBin MCTS GRPO
- ☐ **Experiment C** (if relevant): Cluster entropy MCTS GRPO
- ☐ Track per-step: train accuracy, eval accuracy (MATH500 test), KL to base, reward variance, advantage variance

### 4.3 What We're Looking For
- ☐ Does tree-based RL achieve same/better final accuracy as flat?
- ☐ Does tree-based RL exhibit policy drift, instability, or other pathologies?
- ☐ Token efficiency: total compute to reach target accuracy
- ☐ Curriculum interaction (Qirong mentioned this is a can of worms — keep separate / acknowledge as limitation)

---

## Part 5: Crown Jewel — BO / Non-Myopic MCTS

The "advanced" method that goes beyond faithfulness for actual gains, while still avoiding the failure modes characterized in Part 1-4.

- ☐ Design: Bayesian optimization over MCTS hyperparameters per-depth
- ☐ Or: non-myopic search that looks ahead before committing to expansion
- ☐ Raul to lead the BO design (his expertise area)
- ☐ Demonstrate: better than naive MCTS baselines AND faithful methods AND flat rollouts

---

## Part 6: Paper Writing

### 6.1 Structure (tentative)
- §1 Intro: claim that MCTS-RL papers cherry-pick, our paper is the careful study
- §2 Empirical characterization of flat rollout trees (Part 1)
- §3 Faithful tree methods (Part 2)
- §4 Baselines and failure mode analysis (Part 3)
- §5 End-to-end RL validation (Part 4)
- §6 Crown jewel: BO MCTS (Part 5)
- §7 Discussion: trade-offs, when to use which method

### 6.2 Figure Budget
| Section | Figures (target) |
|---------|-----------------|
| §2 Characterization | 3-4 (WBSP curves with std, distribution fits, depth-purity) |
| §3 Faithful methods | 2-3 (method diagrams, accuracy comparison) |
| §4 Baselines | 2 (4-method comparison, failure mode examples) |
| §5 RL validation | 3-4 (training curves, eval accuracy, drift metrics) |
| §6 Crown jewel | 2-3 (BO MCTS results) |

---

## Immediate Next Steps (Priority Order)

1. **[HIGH] Split "no-advantage" into all-correct vs all-wrong** for the 100-problem analysis
   - Update `poisson_mcts/plot_advantage_comparison.py`
   - Re-generate purity_analysis.png and summary.png
   - Update README with new terminology

2. **[HIGH] Rename Poisson MCTS → NegBin MCTS** across codebase
   - Files: `poisson_mcts_engine.py`, `poisson_mcts_tree.py`, output dirs
   - Plots and READMEs

3. **[HIGH] Add std bars to WBSP / branching curves**
   - Update plots in `figures/training_evolution/`, `figures/branching_analysis/`

4. **[MEDIUM] Split WBSP plots by problem difficulty** (easy/hard buckets)

5. **[MEDIUM] Add 3rd candidate distribution** for branching factor fitting + statistical comparison

6. **[MEDIUM] Implement cluster entropy MCTS baseline**

7. **[MEDIUM] Investigate non-parametric distribution fitting** (Raul's suggestion)

8. **[LOW for now] Plan end-to-end RL training infrastructure** — needs sustained GPU allocation

---

## Open Questions / Decisions Needed

- Do we include curriculum design experiments? (Qirong: "can of worms, probably leave for later")
- How many baselines is enough? Is cluster entropy alone sufficient, or do we also need PRM-based?
- Should we extend to step_40/80/120 stages (we currently only have 100-problem comparison on step_0)?
- Final problem count: 100 sufficient for current results, but for paper we may want all 400 MATH500 train problems
- Which RL framework: verl, OpenRLHF, or custom? Need to pick before Part 4

---

## Meeting Notes Archive

- **2026-04-09**: Weekly sync with Qirong & Raul. Decisions captured in this doc. Key surprise: faithful methods have **higher** accuracy than flat (intriguing, needs careful framing). Pure problems analysis flagged as misleading without correct/wrong split.
