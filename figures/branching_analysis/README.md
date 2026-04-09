# Branching Analysis

本目录包含分支因子分析和树结构特征的可视化图表。

## 图表说明

### `bf_comparison_all_k.png`
**不同 k 设置下的分支因子对比。** 多面板图，比较不同 num_children（k）设置下的分支因子分布，展示 k 的选择如何影响树的结构。

### `branching_factor_distributions.png`
**分支因子分布。** 以箱线图/小提琴图的形式展示不同深度和训练阶段下分支因子的分布情况，揭示分支因子的集中趋势和离散程度。

### `d0_bf_histograms.png`
**D0 分支因子详细直方图。** 聚焦于深度 0 的分支因子分布，清晰展示 D0 的过度离散现象（方差远大于均值），这是选用 NegBin 而非 Poisson 分布的实证依据。

### `max_depth_distribution.png`
**最大深度分布。** 展示推理路径所达到的最大深度的分布情况，反映模型推理链的长度特征。

### `tree_curves_WBSP.png`
**WBSP 树结构曲线。** 展示 Width（宽度）、Breadth（广度）、Survival（存活率）、Path-length（路径长度）四个指标随深度变化的曲线，全面刻画树的结构特征。
