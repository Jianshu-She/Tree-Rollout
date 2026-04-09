# Distribution Fitting

本目录包含分布拟合相关的可视化图表。我们对 MCTS 树的分支因子（branching factor）和节点准确率（node accuracy）进行了系统的分布拟合分析。

## 图表说明

### `hero_distribution_choice.png`
**核心结论图 — 分布选择策略。** 展示了我们的关键发现：深度 0（D0）的分支因子服从 NegBin 分布（过度离散），而 D1+ 的分支因子服从 Poisson 分布。图中将拟合曲线叠加在经验直方图上，直观地展示了拟合质量。

### `negbin_vs_poisson_d0.png`
**D0 处 NegBin vs Poisson 对比。** 直接比较两种分布在深度 0 的拟合效果，证明 NegBin 能更好地捕捉 D0 的过度离散特征（variance > mean），而 Poisson 在 D0 拟合不佳。

### `parameter_heatmaps.png`
**参数热力图。** 以深度 × 训练阶段（step_0 到 step_120）的网格形式展示所有拟合参数（λ, r, p, α, β），可以一目了然地看到参数随深度和训练进程的变化趋势。

### `beta_d0_illustration.png`
**D0 节点准确率的 Beta 分布示意图。** 展示了深度 0 处节点级别准确率的 Beta 分布拟合，说明了 Beta(α, β) 如何描述节点正确率的分布特征。

### `beta_fit_histograms.png`
**Beta 分布拟合直方图。** 在多个深度处展示节点准确率的经验直方图，并叠加 Beta 分布拟合曲线，验证 Beta 分布对节点准确率建模的合理性。

### `parameter_curves.png`
**参数演化曲线。** 展示拟合参数随深度变化的趋势曲线，不同训练阶段（step_0, step_40, step_80, step_120）用不同颜色表示，揭示 RL 训练如何影响树的结构参数。

### `parameter_summary.png`
**参数汇总。** 所有拟合参数的汇总表/可视化，提供全局概览。

### `poisson_fit_histograms.png`
**Poisson 拟合直方图（D1+）。** 展示 D1 及更深层的分支因子直方图，并叠加 Poisson PMF 拟合曲线，验证 Poisson 分布在 D1+ 的拟合质量。

### `poisson_vs_negbin_histograms.png`
**Poisson vs NegBin 多深度对比直方图。** 在多个深度上并排展示 Poisson 和 NegBin 的拟合效果，进一步佐证"D0 用 NegBin、D1+ 用 Poisson"的分布选择策略。
