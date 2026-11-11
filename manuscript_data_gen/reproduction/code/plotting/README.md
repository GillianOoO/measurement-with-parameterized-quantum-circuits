# Plotting

- `plot_figure_3.py` 读取 `data/shared_processed/figure3/empirical50/fig3_empirical_curves.csv` 及其校验 manifest 作为误差/噪声数据，读取 `fig3_extended_curves.csv` 作为所需采样数投影，生成六子图 Figure 3。
- `plot_figures_2_4_5_and_supp.py` 读取分类后的 H4/H6、resource/variance 与 random-Hamiltonian 表格，生成 Figures 2、4、5 和补充材料方差图。
- `random_variance_ensemble50.py` 核对100个Hamiltonian与共同态的hash、4200条完整网格记录、精确方差和MSE恒等式，再按50实例平均（包含零方差实例）。Main Figure 5委托 `../experiments/plot_random_ensemble.py` 独立复算210000次估计，数据根为 `data/random_ensemble_50/`。
- `plotting_backend.py` 统一字体、颜色、marker、panel label、完整边框和分类后数据路径。

默认输出为 `../../figures/rebuilt/`；`../../figures/published/` 保存最新稿件使用的版本；Main 5/SI 1已换成50实例原始资产，非旧5实例图。

完整、可移植流程见 [`../../FIGURE_REPRODUCTION.md`](../../FIGURE_REPRODUCTION.md)。
推荐使用归档根目录的 `reproduce_manuscript_figures.py --output build/current_figures`，
将验证输出与正式文件隔离。`--figures s1` 可在空目录单独重绘 SI，无需旧 manifest。
每次报告仅包含本次执行的图，不继承历史 PASS 状态。

## SI Figure 1: variance and absolute bias (50 instances, 2026-11-10)

`plot_figures_2_4_5_and_supp.py --supplement-variance-only` 只重绘 SI Figure 1，不重绘正文 Figure 4。当前 SI 图为六个子图，每个子图包含上方的方差曲线和下方从零向下延伸的倒置 bias 柱状图；二者共享对数横轴，下方使用独立缩放的左侧线性纵轴。内部绘图坐标为负的 absolute bias，柱长和刻度标签仍表示非负绝对值，不表示 signed bias。

- Molecular SRDD bias: `data/shared_processed/variance/supp_state_dependent_variance_by_budget.csv` 的 `abs(signed_approximation_bias_hartree)`；使用各 T 已选择的分解，不使用有限次测量的 empirical bias。
- Sparse/dense GPD bias: `data/random_ensemble_50/cases/{sparse,dense}4_seed0_{1,...,50}/gpd_selected/selected_results.csv` 的 `bias`。先对各实例取绝对值，再对50 个实例求均值。代码核对 `bias = approximate_expectation - target_expectation`。
- Bias 绘图数值输出：`data/shared_processed/presentations/supp_absolute_bias_by_budget.csv`，当前 38 行，包含 panel、benchmark、method、measurements、absolute_bias、instances 和 averaging rule。SI (e,f) 对全部方法和 bias 柱统一保留稿件 T>=45 的显示区间，均不显示 T=12。原始模拟文件及完整 random variance summary 保留 T=12。
- 组合图：`figures/rebuilt/supp_state_dependent_variance_vs_budget.pdf` 和 `.png`；六张独立子图：`figures/rebuilt/panels/supp_variance_bias_panel_[a-f].png`。
- 当前发布副本：组合图位于 `figures/published/`，独立子图位于 `figures/panels/`；私有 TeX 不在公开仓库中。重绘仅写 rebuilt，需核对后同步发布副本。
- 复核入口：`validation/validate_bias_update.py --plot-only --rebuild-dir build/current_figures`，检查当前 SI 倒置柱图的数值、零基线、方向、左侧坐标轴及独立重绘。详见归档根目录 `FIGURE_REPRODUCTION.md`。
