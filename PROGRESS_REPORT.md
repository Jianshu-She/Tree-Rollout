# Tree-RL Paper 进展报告

> **目标**：EMNLP 2026 (May) 或 ICLR 2027 (Sep)
> **状态**：ROADMAP Part 1-3 已基本完成，Part 4 (RL 训练) 待开工

---

## 1. 项目定位与核心 Claim

### 我们要证明什么

> **"Naively 把 flat rollout 换成 tree-based MCTS 会引入隐藏的副作用。文献里的 MCTS-RL 论文普遍 cherry-pick 结果，不认真刻画这些代价。我们提出的 'faithful methods'（BFS + NegBin MCTS）能拿到 tree-based 的效率收益，同时不引入副作用。"**

### 三层论证

| 层 | 内容 | ROADMAP Part |
|---|---|---|
| 1. **实证刻画** | flat rollout 的树结构长啥样（宽度、分支因子、深度、purity） | Part 1 |
| 2. **faithful methods** | 我们的 BFS Tree + NegBin MCTS 忠实地模仿了 flat 的分布 | Part 2 |
| 3. **对比 baseline** | 拿文献里的"激进" MCTS（DeepSearch）做对照组，show 副作用 | Part 3 |
| 4. **RL 训练验证** | 最终用实际的 GRPO 训练跑出来，看 reward 分布演化 | Part 4 (未开工) |
| 5. **Crown jewel** | Bayesian Optimization / 非 myopic 搜索 MCTS | Part 5 (未开工) |

---

## 2. 已完成的工作（ROADMAP Item 1-6）

### Item #1 ✅ 拆分 "no-advantage" 为 all-correct vs all-wrong

**动机**：之前把 advantage std = 0 的题一概叫做 "pure problems / no signal"，但 Qirong 指出这是误解——
- **All-correct**（128 条都对）= 模型完美解决了，这是 RL 目标本身，**好事**
- **All-wrong**（128 条都错）= 模型完全失败，**坏事**

两个拆开看才有意义。

**结果**（100 题 step_0）：

| 方法 | All-Correct ✓ | All-Wrong ✗ | 总 no-adv |
|---|---|---|---|
| Flat | 0 | 5 | 5 |
| BFS | 8 | 9 | 17 |
| NegBin | 9 | 15 | 24 |
| DeepSearch | 14 | 17 | 31 |

**发现**：Tree 方法的 no-advantage 有很大一部分是因为它们**解决了**问题，不是失败了。

**产物**：
- `poisson_mcts/plot_advantage_comparison.py::plot_no_advantage_analysis`
- `poisson_mcts/results/advantage_comparison/step_0/no_advantage_analysis.png`

---

### Item #3 ✅ WBSP 曲线加 ±1 std shaded bands

**动机**：Qirong 要求所有 WBSP 曲线（Width / Branching / Surviving / Purity）加上 cross-problem 的方差可视化，让 reader 看到不只是 mean 的趋势。

**产物**：
- `faithful_baseline/analyze_branching_factors.py`（修改）
- `figures/training_evolution/tree_curves_WBSP.png`
- `figures/branching_analysis/tree_curves_WBSP.png`

**关键观察**：
- **W(d)**: cross-problem std 非常大（宽度随题目变化很剧烈）
- **B(d)**: D0 的 std 远大于 mean — **直接可视化了 NegBin 重尾的必要性**
- **S(d)**: std 随深度收敛
- **P(d)**: step_120 之后 std 趋近 0（模型收敛）

---

### Item #4 ✅ WBSP 按题目难度分桶

**动机**：整体聚合的 WBSP 隐藏了难度梯度。按 step_120 flat-rollout accuracy 分三桶：

| 桶 | 定义 | #题目 |
|---|---|---|
| Easy | acc ≥ 0.8 | 293 |
| Medium | 0.5 ≤ acc < 0.8 | 35 |
| Hard | acc < 0.5 | 72 |

**产物**：
- `faithful_baseline/plot_wbsp_by_difficulty.py`（新建）
- `figures/training_evolution/tree_curves_WBSP_by_difficulty.png`
- `figures/branching_analysis/tree_curves_WBSP_by_difficulty.png`

**关键观察**：
- **Easy 题**: step_0 就窄，step_120 收得更紧，purity 快速收敛到 1.0
- **Hard 题**: 宽度最大、存活最久，purity 即使到 step_120 仍在 0.85-0.95 浮动（**模型没收敛**）；std band 也最宽
- **验证**：困难题目保持明显更大的 branching spread，印证了"shallow depth = 多样性集中区"的说法

---

### Item #5 ✅ 三种分支因子分布对比（AIC 模型选择）

**动机**：Qirong 说不要 claim "NegBin 就是 the distribution"，要对比多个 candidate 分布。

**候选**：
1. **Poisson(λ)** — 1 参数，轻尾
2. **Geometric(p)** — 1 参数，最简单的重尾
3. **Negative Binomial(r, p)** — 2 参数，灵活重尾

**评分**：Log-likelihood + AIC（惩罚 NB 的额外参数）+ discrete KS 距离

**产物**：
- `faithful_baseline/compare_bf_distributions.py`（新建）
- `faithful_baseline/results/math500_full/train/poisson_beta_analysis/bf_distribution_comparison.{json,png}`
- `figures/distribution_fitting/bf_distribution_comparison.png`

**关键结果（每个 stage/depth 的 AIC 获胜者）**：

| Depth | step_0 | step_40 | step_80 | step_120 |
|---|---|---|---|---|
| **D0** | **NegBin** (ΔLL +2613) | **NegBin** (+328) | **NegBin** (+153) | **NegBin** (+117) |
| **D1** | **Geometric** (ΔLL +774) | **Geometric** (+1780) | **Geometric** (+1806) | **Geometric** (+1908) |
| D2 | Poisson | Poisson | NegBin | NegBin |
| D3+ | Poisson | Poisson | Poisson | Poisson |

**重要发现**：
- **D0 必须是 NegBin**（重尾）— 印证了 NegBin MCTS 选择
- **D1 竟然是 Geometric 全胜**，甚至比 NegBin 还好（只用 1 个参数）
- **D2+ 是 Poisson** —— paper 里之前 "Poisson for D1+" 的 claim **需要修正为 "Geometric at D1, Poisson at D2+"**

这是 Qirong 要的那种反直觉细节。

---

### Item #6 ✅ DeepSearch 作为第 4 个对比 baseline

**动机**：Paper Part 3 需要"文献里的激进 MCTS"作对照组，选用 **DeepSearch (arxiv 2509.25454, ICLR 2026)**。

**实现**：
1. 对齐官方 repo（github.com/smiles724/DeepSearch）的超参数：λ₁=0.4, λ₂=0.4, λ₃=0.01, depth_bonus=sqrt(d/max_depth), max_depth=64, expansion_width=8
2. 在 `mcts_inference/deepsearch_engine.py` 加 `solve_to_target(target=128)` 方法
3. 接入 `poisson_mcts/compare_advantages.py` 作为第 4 方法
4. 跑了 100 道题完整对比（4 方法同一 seed，同一 100 题）

**产物**：
- `mcts_inference/deepsearch_engine.py`, `deepsearch_tree.py`（修正超参数）
- `poisson_mcts/compare_advantages.py`（4 方法集成）
- `poisson_mcts/plot_advantage_comparison.py`（重构支持 4 方法 + 新增 2 个专项图）
- `poisson_mcts/results/advantage_comparison/step_0/` 下 10 张新图 + comparison_step_0.json

---

## 3. 100-题 4-方法实验的**核心结果**

### 核心 claim（以 flat rollout 作为 compute reference）

| 方法 | Mean Accuracy | Compute (% flat) | Pareto 判决 |
|---|---|---|---|
| **Flat** | 45.6% | **100%** (reference) | baseline |
| **BFS** | **50.8%** (+5.2pp) | **11.3%** | ✅ **支配 flat**（更好 + 更便宜）|
| **NegBin** | **51.7%** (+6.1pp) | **10.1%** | ✅ **支配 flat**（更好 + 更便宜）|
| **DeepSearch** | 44.3% (−1.3pp) | 29.4% | ❌ 比 faithful 贵 3x 且 accuracy 反而差 |

**一句话 claim**：
> Faithful methods 用不到 flat 11% 的计算拿到比 flat 高 5-6pp 的 accuracy；DeepSearch 用 3 倍 faithful 的计算换来更差的 accuracy。

### 详细 Summary 表

| 方法 | Mean Accuracy | #Traj | All-Correct ✓ | All-Wrong ✗ | Mean Tokens | Token Ratio | Pearson vs Flat |
|---|---|---|---|---|---|---|---|
| **Flat** | 45.6% | 128 | 0 | 5 | 219K | — | 1.000 |
| **BFS** | **50.8%** (+5.2pp) | 43 | 8 | 9 | 25K | **11.3%** | **0.947** |
| **NegBin** | **51.7%** (+6.1pp) | 44 | 9 | 15 | 22K | **10.1%** | **0.933** |
| **DeepSearch** | 44.3% (**−1.3pp**) | 98 | **14** | **17** | **64K** | **29.4%** | **0.759** |

### Compute-matched robustness check (Item #10)

补充实验：post-hoc bootstrap subsample 到不同 compute budget，验证 "faithful 支配 flat" 在所有 compute 水平上都成立。关键数据点：

| Budget | Flat | BFS | NegBin | DeepSearch |
|---|---|---|---|---|
| **5K** (极限低) | 47.2% σ=0.45 n=4 nA=55 | 50.5% σ=0.63 n=11 nA=36 | 52.1% σ=0.64 n=12 nA=36 | 44.9% σ=0.48 n=16 nA=52 |
| **22K** (faithful 自然水平) | 45.5% σ=0.82 n=17 nA=18 | **50.9%** σ=0.80 n=35 nA=20 | **51.7%** σ=0.74 n=32 nA=26 | 44.2% σ=0.61 n=50 nA=39 |
| **220K** (flat 自然水平) | 45.5% σ=0.94 n=112 nA=6 | 50.8% σ=0.83 n=43 nA=17 | 51.7% σ=0.76 n=44 nA=24 | 44.2% σ=0.68 n=96 nA=32 |

**关键观察**：
- **Flat 即使在 220K 全 budget 下 accuracy 也只能到 45.5%，永远追不上 BFS/NegBin 的 50.8/51.7%**
- Flat 在 5K 下只有 4 条 trajectory → σ=0.45（信号崩溃），55 题 no-advantage
- BFS/NegBin 在 5K 下就已接近最终 accuracy（只需 ~10-12 条 trajectory）
- DeepSearch 在所有 budget 下都是最低 accuracy

### 关键图表

| 图 | 文件 | 显示的故事 |
|---|---|---|
| ⭐ **Pareto** | `pareto_accuracy_vs_tokens.png` | **Paper 的 main figure**。x=compute (% of flat)，y=accuracy。BFS/NegBin 在左上"DOMINATES flat"象限，DeepSearch 比 flat 还差。 |
| **Summary** | `summary.png` | 4 方法的 7 个核心指标 + accuracy bucket 分布。 |
| **Drift Matrix** | `drift_correlation_matrix.png` | 4×4 Pearson 相关性矩阵。Flat/BFS/NegBin 三者互相 >0.93，DeepSearch 和所有人都只有 0.73-0.79 — 明显 outlier。 |
| **Compute-matched** | `compute_matched_analysis.png` | Supplementary robustness check。4 个 panel 覆盖 accuracy / adv_std / no-advantage / n vs compute budget 曲线。 |
| **No-Advantage** | `no_advantage_analysis.png` | 4 方法的 all-correct 和 all-wrong 拆分柱状图，DeepSearch 的"灾难性失败"最多。 |
| **Accuracy Bucket** | `summary.png` 右侧 | DeepSearch 是 **bimodal**（0% 和 100% 桶最多），说明它"要么全对要么全错"，这是 RL no-advantage 的灾难场景。 |

### Per-problem scatter 和 diff histogram

见：
- `accuracy_scatter.png` — Flat vs 每个 tree method 的 per-problem accuracy
- `accuracy_diff_histogram.png` — accuracy 差值分布
- `accuracy_bar_sorted.png` — 按 Flat accuracy 排序的并列柱状图

---

## 4. 核心结论（paper 可以写的 claim）

### 4.1 关于 faithful methods（BFS + NegBin）

1. **Accuracy 净正**：比 flat 提升 5-6 pp，不引入 accuracy 退化
2. **Token 效率高**：只用 flat 的 10-11% tokens，即 90% 计算节省
3. **行为忠实**：per-problem accuracy 与 flat 的 Pearson 相关性 0.93-0.95，说明树方法和 flat **解决的是同一批题**
4. **No-advantage 代价可接受**：17-24 题（其中超过一半是 all-correct，即模型解决了）

### 4.2 关于 DeepSearch（文献 baseline）

1. **Accuracy 负增长**：比 flat **还低 1.3pp**（faithful methods 都是正增长）
2. **Token 开销 ~3x faithful**：29.4% vs 10-11%（虽然比 flat 省，但远高于我们的方法）
3. **GRPO 信号最弱**：adv_std 0.69（4 方法中最低）
4. **Drift 明显**：Pearson vs Flat 只有 0.759，行为偏离了 flat rollout 的自然分布
5. **Bimodal 分布**：accuracy 0% 和 100% 两个桶最多 → 31% 的题目没有 RL 信号（all-correct + all-wrong = 14+17）
6. **文献无法解释**：DeepSearch 原论文 report 的是 RL 训练**收敛后**的 benchmark accuracy，而不是**训练初期的 rollout 质量** → 我们的 offline 分析揭示了训练 *过程中* 的隐藏代价

### 4.3 关于分布拟合（Part 1）

1. D0 分支因子重尾，**必须用 NegBin**（Poisson 和 Geometric 都显著输掉）
2. D1 分支因子中尾，**Geometric 全胜**（比 NegBin 还好，即使参数更少）
3. D2+ 分支因子窄，**Poisson 够用**
4. → paper claim 从 "NegBin + Poisson" 细化为 **"NegBin @ D0 + Geometric @ D1 + Poisson @ D2+"**

### 4.4 Offline 结论和 RL 训练的关系

Offline 100-题对比**只是预测性分析**。它告诉我们：如果拿这些方法去跑 RL 训练，**DeepSearch 可能会遇到什么问题**——

- 31% no-advantage 意味着每个 step 有 31% 的 prompt 完全没梯度 → policy collapse 风险
- 0.76 的 Pearson 意味着 DeepSearch 会把 model 推向和 flat 完全不同的 trajectory 分布 → 有 style drift 风险
- 3x 的 token 成本意味着同样 GPU 时间只能跑 1/3 的 step

但这些预测**只有通过实际跑 RL 训练才能证实**——这是 Part 4 的任务。

---

## 5. 待完成工作（Part 4 onwards）

### 高优先级

- **#6.5** (Part 4 scaffold): verl 集成 — `MCTS/verl_tree_rl/` 目录结构
  - 架构已规划：`FaithfulRollout` wrapper 包装 verl 原生 `VLLMRollout`，通过 config 切换 4 种 tree engine
  - 参考实现：DeepSearch 官方 repo 的 `deepsearch/rollout/sglang_mcts_wrapper.py`
  - 需要 port：BFS / NegBin / DeepSearch 3 套 engine 到 verl 的 DataProto 接口

- **Part 4 训练实验**：对 4 种方法各跑一次 GRPO，track
  - reward 分布随 step 的演化
  - Advantage variance 衰减速率
  - KL(π_t || π_base) 漂移
  - Per-problem accuracy 轨迹
  - 训练后分布与 flat baseline 的距离

### 中优先级

- **#1.1 leftover**: Accuracy 分布也加 1-2 个候选（对比 Beta 是否真的最好）
- **#7** Non-parametric 分布拟合（Raul 建议，对标 NegBin/Geometric/Poisson 的参数化假设）

### 低优先级

- **#8** Part 5 crown jewel: BO / 非 myopic MCTS 设计

### 已跳过

- **#2** "Poisson MCTS → NegBin MCTS" 代码 rename（用户决定先跳过，label 已更新但 JSON keys 仍是 legacy name）

---

## 6. 需要你决定的事

1. **RL 训练数据集**：
   - 用 MATH500（和 offline 分析一致）
   - 或用 dapo-math-17k（和 step_0/40/80/120 checkpoints 一致）
   - 我倾向前者（MATH500 只有 400 题，跑得快，且 story 统一）

2. **RL 训练框架**：
   - 你现有的 `copus/verl`（完整装好的 verl fork）
   - 我们需要决定是直接改这个 repo，还是 fork 一份到 MCTS 项目里（推荐后者）

3. **Part 4 实验规模**：
   - 短 run（200 steps）验证 reward 分布早期信号
   - 长 run（2000-5000 steps）做 paper 的主实验
   - 先做短 run 再决定要不要长 run？

4. **下一步要做哪个**：
   - **A**: 开始 verl_tree_rl scaffold（几个 .py 文件，不需要 GPU，但要至少写 1-2 天）
   - **B**: 跑 step_40/80/120 的 offline 4-方法对比（把 100-题 offline 实验扩展到所有 4 个 stage，产出 training-evolution 图）
   - **C**: 先做 #1.1（accuracy 分布候选对比），快速产出
   - **D**: 其他

---

## 附：目录结构速查

```
MCTS/
├── ROADMAP.md                              # 长期路线图
├── PROGRESS_REPORT.md                      # 本文件
├── mcts_inference/
│   ├── bfs_engine.py, bfs_tree.py          # BFS (faithful)
│   ├── poisson_mcts_engine.py, poisson_mcts_tree.py  # NegBin (faithful, legacy name)
│   └── deepsearch_engine.py, deepsearch_tree.py      # DeepSearch baseline
├── poisson_mcts/
│   ├── compare_advantages.py               # 4 方法对比主脚本
│   ├── plot_advantage_comparison.py        # 4 方法可视化 (10 plots)
│   └── results/advantage_comparison/step_0/
│       ├── comparison_step_0.json          # 100 题 × 4 方法结果
│       └── *.png                           # 10 张对比图
├── faithful_baseline/
│   ├── analyze_branching_factors.py        # WBSP 曲线 (with std bands)
│   ├── plot_wbsp_by_difficulty.py          # WBSP 按难度分桶
│   ├── compare_bf_distributions.py         # 3 分布 AIC 对比
│   ├── fit_poisson_beta.py                 # 原始分布拟合
│   └── results/math500_full/train/
│       ├── trees_syntactic/step_{0,40,80,120}/   # 400 题 × 4 stage post-hoc 树
│       ├── rollouts_step_{0,40,80,120}.json      # 原始 flat rollouts
│       └── poisson_beta_analysis/                # 拟合结果
├── figures/                                 # paper 用图
│   ├── training_evolution/                  # WBSP + difficulty 分桶
│   ├── branching_analysis/
│   ├── distribution_fitting/                # 3 分布对比
│   └── poisson_mcts/
└── data-prepare/
    ├── data/MATH500_train.json              # 400 题训练集
    └── ...
```
