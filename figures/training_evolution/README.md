# Training Evolution

本目录包含 RL 训练过程中树结构演化的可视化图表，追踪从 step_0（基础模型）到 step_120 的变化。

## 图表说明

### `rl_training_evolution.png`
**RL 训练演化多面板图。** 展示 Poisson λ、准确率、深度等关键指标如何随 RL 训练步数（step_0 → step_40 → step_80 → step_120）演化，揭示训练对推理树结构的影响。

### `evolution_plots.png`
**额外演化分析图。** 多面板补充图表，从更多维度展示训练阶段之间的变化趋势。

### `tree_curves_WBSP.png`
**训练阶段间 WBSP 曲线对比。** 将不同训练阶段的 Width、Breadth、Survival、Path-length 曲线绘制在一起，直观对比训练如何改变树的结构特征。
