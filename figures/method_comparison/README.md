# Method Comparison

本目录包含不同搜索方法之间的对比可视化图表，主要比较 MCTS、DeepSearch 和 Flat Rollout。

## 图表说明

### `bf_deepsearch_vs_flat.png`
**DeepSearch vs Flat Rollout 分支因子对比。** 展示 DeepSearch 方法和 Flat Rollout 方法的分支因子剖面差异，揭示两种方法在树结构上的不同。

### `branching_profile_comparison.png`
**多方法分支剖面对比。** 将多种方法的分支因子剖面绘制在一起，全面比较不同搜索策略产生的树结构差异。

### `mcts_vs_flat_accuracy.png`
**MCTS vs Flat Rollout 准确率散点图。** 每个点代表一道题，X 轴为 Flat Rollout 准确率，Y 轴为 MCTS 准确率，展示两种方法在逐题准确率上的对应关系。

### `mcts_vs_flat_lambda.png`
**MCTS vs Flat Rollout Poisson λ 对比。** 比较 MCTS 树和 Flat Rollout 树拟合得到的 Poisson λ 参数，分析两种方法在分支因子统计特征上的异同。
