# Code map

## Methods

- `methods/gpd/`：周期 iSWAP GPD 残差分解、前缀生成以及基于 stabilizer calibration states 的 fixed-number-of-measurements 重选择。
- `methods/srdd/`：RC-DF 张量拟合、受限深度 orbital rotations、F3–R2 redistribution、shallow one-body completion 与校准/选择工具。
- `methods/pauli_product/`：OGM、SG、Derand、LCS 和 AP 的 product-Pauli measurement schedules 与有限测量重放。
- `methods/fully_commuting/`：FC-IMA 的 fully commuting grouping、线路综合、迭代 measurement allocation、误差与噪声评估。

## Experiments and analysis

- `experiments/h4_bond_scan_srdd.py` 与 `h4_bond_scan_pauli.py`：图 2(a,b) 的相同 H4 Hamiltonian 输入。
- `experiments/molecular_srdd_rank_fit.py`、`molecular_srdd_selection.py`、`large_molecule_srdd_pauli.py`：H4/H6/BeH2/N2 的 SRDD 与 Pauli 基线流程。
- `methods/gpd/generate_frobenius_prefixes.py`、`reselect_stabilizer_balanced.py` 和 `methods/pauli_product/replay_random_all_methods_empirical50.py`：图 5 的 GPD 与五种 Pauli 基线。
- `analysis/`：从已审计方法结果生成图 3、图 4 和补充材料所需的派生数据。
- `plotting/plot_figure_3.py`：图 3；`plotting/plot_figures_2_4_5_and_supp.py`：图 2、4、5 和补充材料方差图。

分类后的绘图脚本只读取 `data/` 下的相对路径并输出到 `figures/rebuilt/`。算法入口保留原始随机种子、收敛条件和 schedule 语义；完整环境依赖以各脚本导入项和数据 manifest 为准，虚拟环境不纳入论文归档。

## Native two-qubit gate accounting

iSWAP is a basic two-qubit gate and is not converted to CNOTs. Periodic GPD
with even n and L entangling layers has 3n(L+1) angles, nL/2 native iSWAP
gates, and entangling depth L per setting. For n=4, L=4 these are 60, 8,
and 4, respectively. Counts exclude state preparation, single-qubit gates,
routing and readout.

`analysis/native_gate_resources.py` records CNOT and iSWAP counts separately;
`analysis/build_resource_statistics.py` carries both through fixed-geometry
and bond-scan summaries. See [analysis/README.md](analysis/README.md) for fields,
legacy open-chain accounting and the status of frozen resource CSVs.
