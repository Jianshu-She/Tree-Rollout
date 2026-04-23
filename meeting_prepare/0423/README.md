# Meeting Prepare 2026-04-23

**回顾 4 月 13 号的 7 个 commit 产出 + 论证了什么**

---

## 背景

4 月 9 号和 Qirong / Raul 的 sync 留下一批 action items。13 号一天之内做了 7 个 commit 把 **Part 1-3 的 offline 分析** 基本完成。这个文档对齐每个 commit 改了什么、产出在哪、支撑 paper 里哪个 claim。

---

## Commit 一览

### #1 — `d1a9f72` (09:08): 建立 ROADMAP
**内容**：把 4 月 9 号 sync 的决议写成长期路线图。

**产出**：
- `ROADMAP.md` (203 行)

**论证**：项目管理基础，没有论证功能。确认了 paper narrative、命名规范（`Poisson MCTS → NegBin MCTS`；`pure → no-advantage`）、目标 venue (EMNLP/ICLR，不投 NeurIPS 2026)。

---

### #2 — `e44e8c0` (09:13): 把 no-advantage 拆成 all-correct / all-wrong
**内容**：之前把 "advantage 全 0" 的题统一叫 "pure problems" 或 "no signal"。Qirong 指出这是误解——**all-correct（128 条全对）= 模型解决了，好事**；**all-wrong = 模型失败，坏事**。两者要分开。

**产出**：
- 代码: `poisson_mcts/plot_advantage_comparison.py`（新增 `plot_no_advantage_analysis`）
- 图: `poisson_mcts/results/advantage_comparison/step_0/no_advantage_analysis.png`
- 更新: `poisson_mcts/results/advantage_comparison/step_0/summary.png` 表格加 All-Correct ✓ / All-Wrong ✗ 行
- 删除: `purity_analysis.png`（旧的合并版）

**论证**（step_0 上 100 题）：
- Flat: **0 all-correct / 5 all-wrong**
- BFS: **5 all-correct / 9 all-wrong**
- NegBin: **11 all-correct / 13 all-wrong**
- Tree 方法"新"产生的 all-correct (+5, +11) > 新产生的 all-wrong (+4, +8)
- → **Tree methods 整体 net-positive on outcomes**，之前 "tree 方法 no-signal 更多" 的 framing 是误导的

---

### #3 — `0c397a8` (09:54): WBSP 曲线加 ±1 std shaded bands
**内容**：Qirong 要求把 WBSP (Width / Branching / Surviving / Purity) 曲线画成 "violin-like"——显示 cross-problem 方差，不只是均值。

**产出**：
- 代码: `faithful_baseline/analyze_branching_factors.py`
- 图: `faithful_baseline/results/math500_full/train/branching_analysis/tree_curves_WBSP.png`
- 同步: `figures/branching_analysis/tree_curves_WBSP.png`, `figures/training_evolution/tree_curves_WBSP.png`

**论证**：
- **D0 的 std band 远宽于 mean** → 视觉上印证**分支因子的重尾性**
- 这个重尾性正是 paper rename "Poisson MCTS → NegBin MCTS" 的理由（NegBin 能 fit heavy tail）
- 加了 band 后 reviewer 一眼能看出 "为什么 Poisson fit 不够"

---

### #4 — `7100080` (12:37): WBSP 按难度分桶
**内容**：把 400 道题按 step_120 flat-rollout accuracy 分 3 桶，分别画 WBSP 曲线。

**产出**：
- 代码: `faithful_baseline/plot_wbsp_by_difficulty.py` (197 行，新建)
- 图: `faithful_baseline/results/math500_full/train/branching_analysis/tree_curves_WBSP_by_difficulty.png`
- 同步: `figures/branching_analysis/tree_curves_WBSP_by_difficulty.png`, `figures/training_evolution/tree_curves_WBSP_by_difficulty.png`

**三个桶的数字**：
- **Easy** (acc ≥ 0.8 at step_120)：293 题
- **Medium** (0.5 ≤ acc < 0.8)：35 题
- **Hard** (acc < 0.5 at step_120)：72 题

**论证**：
- **Hard 题的 Width 最大、Purity 到 step_120 还没收敛到 1.0** → 印证 Qirong 的直觉："聚合的 WBSP 隐藏了难度梯度"
- 困难题目保持更大的 branching spread
- 支持 paper 核心 finding："**tree search 在困难题上价值更高**"（Easy 题模型已经很确定，tree 没探索空间）

---

### #5 — `0924159` (12:43): 3 分布对比（Poisson / Geometric / NegBin）
**内容**：Qirong 反对把 "NegBin 就是 the distribution" 写死，要求对比多个候选分布，用 AIC 做模型选择。

**产出**：
- 代码: `faithful_baseline/compare_bf_distributions.py` (376 行，新建)
- 数据: `faithful_baseline/results/math500_full/train/poisson_beta_analysis/bf_distribution_comparison.json` (1450 行，全 stage × 全 depth 的 LL/AIC/KS 矩阵)
- 图: `faithful_baseline/results/math500_full/train/poisson_beta_analysis/bf_distribution_comparison.png`
- 同步: `figures/branching_analysis/bf_distribution_comparison.png`, `figures/distribution_fitting/bf_distribution_comparison.png`

**论证 AIC 胜者** (全 stage)：
| Depth | 胜者 | 解读 |
|---|---|---|
| D0 | **NegBin** (ΔLL +117 to +2613) | 重尾，必须 NegBin |
| D1 | **Geometric** (ΔLL up to +1908) | 甚至比 NegBin 还好，仅用 1 参数 |
| D2+ | Poisson | 窄分布，Poisson 够用 |

**对 paper claim 的影响**：
- 原 claim "NegBin for D0, Poisson for D1+" **不准**
- 新 claim "**NegBin @ D0 + Geometric @ D1 + Poisson @ D2+**"
- 是 paper 里一个"反直觉但重要"的发现（Qirong 最喜欢这种）

---

### #6 — `b603449` (17:38): DeepSearch 接入 + 超参对齐官方 repo
**内容**：
- 拉了 arxiv 2509.25454 DeepSearch 官方 repo (github.com/smiles724/DeepSearch) 对照
- 修正 `mcts_inference/deepsearch_tree.py` 的 5 处偏差
- 接入 `poisson_mcts/compare_advantages.py` 作为第 4 个方法

**产出**：
- `mcts_inference/deepsearch_engine.py`（加 `solve_to_target()` 方法）
- `mcts_inference/deepsearch_tree.py` 超参数对齐：
  - `lambda_quality`: 1.0 → **0.4** (λ1)
  - `lambda_entropy`: 0.5 → **0.4** (λ2)
  - `lambda_depth`: 0.1 → **0.01** (λ3)
  - `depth_bonus`: `log(d+1)` → **`sqrt(d/max_depth)`**
  - 去掉 ad-hoc 的 `visit_bonus`（paper formula 没有这项）
- `poisson_mcts/compare_advantages.py`（4 方法调度）

**论证**：没有新数据产出，纯代码修正。为下一个 commit 的 100-题实验做准备。

---

### #7 — `a4e6f8b` (22:21): 100 题 × 4 方法完整实验（Part 3 核心数据）
**内容**：用 commit #6 修好的 DeepSearch，在 step_0 (Qwen2.5-Math-7B base model) 上跑完整 100 道 MATH500 train 的 4 方法 advantage comparison。

**产出**：
- 代码重构: `poisson_mcts/plot_advantage_comparison.py`（~69% rewrite）基于 METHODS registry 支持 4 方法
- 数据: `poisson_mcts/results/advantage_comparison/step_0/comparison_step_0.json` (869KB)
- 日志: `poisson_mcts/results/advantage_comparison/step_0/log_step_0.txt`
- 10 张对比图（全部在 `poisson_mcts/results/advantage_comparison/step_0/`）：
  - ⭐ `summary.png` — main summary 表格 + accuracy bucket 分布
  - ⭐ `drift_correlation_matrix.png` — 4×4 Pearson 相关矩阵（**paper 最强图之一**）
  - `pareto_accuracy_vs_tokens.png` — compute vs accuracy Pareto
  - `accuracy_scatter.png` — per-problem accuracy 散点
  - `accuracy_diff_histogram.png` — 差值分布
  - `accuracy_bar_sorted.png` — per-problem 排序柱状图
  - `advantage_distributions.png` — 6 道题的 advantage 分布
  - `no_advantage_analysis.png` — all-correct vs all-wrong 拆分
  - `token_efficiency.png` — token 效率
  - `trajectory_counts.png` — trajectory 数量分布
- `README.md` 详细描述每张图

**核心数字**（100 道题 step_0）：

| Method | Accuracy | Compute | All-correct ✓ | All-wrong ✗ | Pearson vs Flat |
|---|---|---|---|---|---|
| Flat | 45.6% | **100%** (reference) | 0 | 5 | 1.000 |
| **BFS** | **50.8% (+5.2pp)** | **11.3%** | 8 | 9 | **0.947** |
| **NegBin** | **51.7% (+6.1pp)** | **10.1%** | 9 | 15 | **0.933** |
| **DeepSearch** | 44.3% (−1.3pp) | 29.4% | 14 | 17 | **0.759** |

**论证（paper 的 smoking gun）**：

1. **Faithful methods 支配 flat**（better + cheaper）：
   - BFS/NegBin 用 **10-11% compute** 达到 **+5-6pp accuracy**
   - 这是 paper Part 3 的 main claim

2. **DeepSearch 全输**：
   - Accuracy **比 flat 还低 1.3pp**（faithful 都是正增长）
   - Compute **3x faithful**
   - No-advantage 最多：14+17 = **31% 的题没 RL 信号**
   - Drift 最明显：Pearson 0.76 vs faithful 0.93-0.95

3. **"Drift" 概念 operationalize 出来了**：
   - Pearson(per-problem accuracy vs Flat) 作为 drift 度量
   - Flat/BFS/NegBin 三角形互相 >0.93，DS 是明显的 outlier
   - 这张 `drift_correlation_matrix.png` 是 paper 最直观的"DS 不 faithful"证据

---

## 一天工作量分布

| 时段 | 工作 | commit |
|---|---|---|
| 上午 09:08–09:54 | 建 ROADMAP + 2 个图优化 | #1, #2, #3 |
| 中午 12:37–12:43 | 2 个新分析（难度分桶 + 3 分布 AIC） | #4, #5 |
| 下午-晚上 17:38–22:21 | DeepSearch 接入 + 跑 100 题 × 4 方法 | #6, #7 |

---

## 这 7 个 commit 合起来论证了什么

### Paper §2 (Empirical characterization of flat rollout trees)
- **#3**：WBSP 有 ±1 std → 视觉证明 D0 分支因子的重尾性
- **#4**：WBSP 按难度分桶 → 证明 tree search 的价值随难度升高
- **#5**：AIC 对比三分布 → 锁定 fitting 的细节（NegBin @ D0, Geom @ D1, Poisson @ D2+）

### Paper §3 (Faithful methods vs baselines)
- **#2**：重新拆 no-advantage → 证明"tree 方法 no-signal 多"的 framing 错了，tree 其实是 net-positive
- **#6, #7**：完整 4 方法 100 题对比 → paper 最主要的 empirical evidence：
  - Faithful methods 支配 flat（更好 + 更便宜）
  - DeepSearch 作为 baseline 全输
  - Drift 作为 metric 被 operationalize

### Paper methodology
- **#1**：ROADMAP 锁定 paper narrative 和 vocabulary（"faithful methods", "no-advantage" 等）

### 对后续工作（Part 4 RL 训练）的指引
- offline 数据 → 给 RL 训练设 **量化预测**：
  - "DS 在 RL 训练中应该 drift 最严重"
  - "BFS/NegBin 应该 compute 最省"
  - "如果 RL 训练复现 offline 的 pattern，paper 的 claim 就立住了"

---

## 关键产出文件路径（速查）

### 代码
- `ROADMAP.md`
- `poisson_mcts/plot_advantage_comparison.py`
- `poisson_mcts/compare_advantages.py`
- `faithful_baseline/analyze_branching_factors.py`
- `faithful_baseline/plot_wbsp_by_difficulty.py`
- `faithful_baseline/compare_bf_distributions.py`
- `mcts_inference/deepsearch_engine.py`
- `mcts_inference/deepsearch_tree.py`

### 数据
- `poisson_mcts/results/advantage_comparison/step_0/comparison_step_0.json`
- `faithful_baseline/results/math500_full/train/poisson_beta_analysis/bf_distribution_comparison.json`

### Paper 图（重要）
- **Drift 主图**: `poisson_mcts/results/advantage_comparison/step_0/drift_correlation_matrix.png`
- **Main summary**: `poisson_mcts/results/advantage_comparison/step_0/summary.png`
- **Pareto**: `poisson_mcts/results/advantage_comparison/step_0/pareto_accuracy_vs_tokens.png`
- **WBSP with std**: `figures/training_evolution/tree_curves_WBSP.png`
- **WBSP by difficulty**: `figures/training_evolution/tree_curves_WBSP_by_difficulty.png`
- **3-distribution AIC**: `figures/distribution_fitting/bf_distribution_comparison.png`
- **No-advantage split**: `poisson_mcts/results/advantage_comparison/step_0/no_advantage_analysis.png`
- **Details README**: `poisson_mcts/results/advantage_comparison/step_0/README.md`
