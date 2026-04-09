# Poisson-MCTS

本目录包含 Poisson-MCTS 方法的实验结果可视化图表，展示其与 Flat Rollout 和 BFS Tree 的对比。

## 图表说明

### `accuracy_comparison.png`
**准确率对比柱状图。** 对比 Flat Rollout、BFS Tree 和 Poisson-MCTS 三种方法的准确率，验证 tree-based 方法能否保持与 flat rollout 可比的准确率。

### `bf_profiles.png`
**Poisson-MCTS 分支因子剖面。** 展示 Poisson-MCTS 生成的树的分支因子随深度变化的剖面，反映分布引导的分支策略实际效果。

### `compute_efficiency.png`
**计算效率指标。** 展示各方法的 token 使用量、运行时间和 token 节省比例，量化 tree-based 方法通过前缀共享带来的计算效率提升。

### `pareto_accuracy_vs_kl.png`
**准确率 vs KL 散度 Pareto 前沿。** 展示不同超参数配置（α, C, temperature）下准确率与 KL 散度的权衡关系，用于超参数选择和分析 Poisson-MCTS 的 Pareto 最优配置。
