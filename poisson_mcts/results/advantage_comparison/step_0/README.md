# GRPO Advantage Comparison: Flat Rollout vs BFS Tree vs Poisson-MCTS

100 道 MATH500 题目，step_0（Qwen2.5-Math-7B 基础模型），目标每种方法产出 128 条完整推理轨迹，计算 GRPO advantage 并对比。

## 图表说明

### `summary.png`
**汇总图。** 左侧为关键指标表格，右侧为正确率分桶柱状图。表格展示三种方法在准确率、轨迹数、纯度（无 RL 信号的题数）、token 开销和相关性上的对比。分桶图揭示树方法将更多题目推向极端（0% 和 100%），中间地带（20-60%）的题目减少。Flat 没有任何题达到 100% 正确率，但 BFS 有 5 题、MCTS 有 11 题达到 100%。

### `accuracy_scatter.png`
**逐题正确率散点图。** 每个点代表一道题，X 轴为 Flat Rollout 正确率，Y 轴为树方法正确率，颜色深浅表示该题生成的轨迹数量。虚线为 y=x 参考线。BFS 与 Flat 的 Pearson 相关性为 0.950，MCTS 为 0.915。大部分点在对角线上方，说明树方法整体正确率略高于 Flat。MCTS 散点更分散（方差更大），部分原因是轨迹数不稳定（颜色深的点偏离更远）。

### `accuracy_diff_histogram.png`
**逐题正确率差值分布直方图。** 左图为 BFS - Flat，右图为 MCTS - Flat。BFS 的差值分布集中在 0 附近，均值 +5.6%，标准差较小，说明 BFS 的行为与 Flat 高度一致且略有提升。MCTS 均值也是 +5.4%，但分布更宽（标准差 14.9%），存在少数差异很大的 outlier（±40%），反映了 MCTS 的随机性更强。

### `accuracy_bar_sorted.png`
**逐题正确率柱状图（按 Flat 正确率排序）。** 100 道题从左到右按 Flat 正确率从低到高排列，三种颜色的柱子并列对比。可以直观看到：简单题（右侧）三种方法表现接近；中等难度题树方法通常更高；困难题（左侧）差异较大且方向不一定。

### `trajectory_counts.png`
**轨迹数分布直方图。** 红色虚线为目标值 128。BFS 的轨迹数集中在 27-52，均值 43，分布非常集中（因为分支因子是确定性的）。MCTS 的轨迹数分布极为分散（1-128），均值 47，说明 MCTS 的树结构因题目而异。两种方法都远未达到 128 条目标，这是导致 purity 升高的根本原因。

### `token_efficiency.png`
**计算效率对比。** 左图箱线图展示三种方法的每题 token 开销：Flat 中位数约 170K token，BFS 和 MCTS 中位数约 25K，约为 Flat 的 12-15%。右图散点图展示逐题的 token 比率（Tree/Flat），BFS 均值 14.9%，MCTS 均值 14.6%。树方法通过前缀共享大幅降低了计算成本。

### `purity_analysis.png`
**Purity 分析：树方法何时丢失 RL 信号？** 左图柱状图展示三种方法的 pure 题目数（adv_std=0，即所有轨迹全对或全错，GRPO advantage 为 0，无训练信号）：Flat 5 题，BFS 14 题，MCTS 24 题。中间散点图展示 MCTS 的轨迹数与 purity 的关系——纯组（红叉）在各个轨迹数量级都存在，并非只出现在轨迹少的题目上，高准确率题（全对）和低准确率题（全错）都容易变纯。右图展示丢失信号的题目在 Flat 中的原始正确率分布：大部分集中在极端值（接近 0% 或 90%+），即本来就接近全对/全错的题目。

### `advantage_distributions.png`
**6 道代表性题目的 GRPO Advantage 分布直方图。** 选取了 6 种典型情况：三方法一致（如 Problem 11 全错）、BFS 大幅优于 Flat、MCTS 大幅优于 Flat、MCTS 大幅劣于 Flat、中等难度题、困难题。每个子图中三种颜色的直方图叠加显示 advantage 值的分布密度，图例标注了正确率和轨迹数。可以看到当正确率不同时，advantage 分布的形状（二值分布的两个 spike 的高度比）会发生偏移。

## 数据文件

### `comparison_step_0.json`
完整的实验结果（869KB），包含 100 道题每种方法的：
- `accuracy`, `adv_std`, `adv_mean`: 正确率和 advantage 统计量
- `total_tokens`, `tokens_per_trajectory`: token 开销
- `num_trajectories`: 实际生成的轨迹数
- `token_savings`: 相对于等价 flat 开销的节省比例
- `flat_advantages`, `bfs_advantages`, `mcts_advantages`: 完整的 advantage 数组（用于分布分析）
