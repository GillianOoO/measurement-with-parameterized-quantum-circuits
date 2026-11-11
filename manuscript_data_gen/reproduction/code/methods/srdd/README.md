# SRDD implementation

- `shallow_rcdf.py`：joint RC-DF tensor fit 与受限深度 orbital rotations。
- `shallow_collector_redistribution.py`：F3–R2 redistribution 与 shallow one-body completion。
- `stabilizer_calibration.py`：通用 stabilizer-calibration、candidate materialization、balanced selection 和 finite-measurement evaluation；当前归档仅保存 H4、H6、BeH2、N2 数据。
- `adaptive_f3_pool.py`、`five_molecule_cases.py`：SRDD 构造使用的共享 fragment/case 工具；当前实验仅调用正文中的四个分子。
- `portable/`：不依赖项目旧目录结构的 SRDD tensor 与 Pauli-completion 核心。

方法输出按分子位于 `../../../data/H4/`、`H6/`、`BeH2/` 和 `N2/`。

Portable entrypoints and verified commands are documented in
[`../../experiments/README.md`](../../experiments/README.md).
`srdd_release_paths.py` resolves the extracted release data without any
machine-specific workspace path. `five_molecule_cases.py` retains its historic
module name for import compatibility but its released dense-case CLI supports
H4/H6; BeH2/N2 use the tensor/MPS entrypoints. `stabilizer_calibration.py` and
`adaptive_f3_pool.py` provide shared SRDD functions; their historical LiH/AGPD
pilot workflows and unrestricted RCDF diagnostic are not release entrypoints.
The `portable/` copies use the common types and utilities from the installed
`mpqc-measurement` package rather than missing relative modules.

## Published molecular protocols

The experiment drivers retain two different calibration and completion
protocols. Both freeze the fitted weights before benchmark-state evaluation.

| Benchmark | Calibration | Shallow one-body completion |
| --- | --- | --- |
| H4/H6, including the H4 bond scan | Equal weight per eligible approximation-rank or sampling-rank/shot stratum; ranks 1–10 and T = 100, 200, 300, 500, 800, 1200, 2000, 3000 | Test added counts in increasing order from 1 through 8; require full numerical row rank and the reconstruction-residual checks |
| BeH2/N2 | Separate nonnegative least-squares fits through the origin for squared bias and sampling variance versus their squared features; all feasible positive candidates and T = 1300, 1600, 2000, 2400, 3000 | Test only one added rotation; accept by the target-reconstruction checks, without the H4/H6 full-row-rank gate |

The H4/H6 one-body-only candidate is excluded from calibration and retained
in selection. The published BeH2/N2 frontiers contain positive factor counts
only. The full-row-rank condition and the one-body-only candidate must not be
imposed on BeH2/N2 when reproducing those experiments.

All four molecular experiments use `collector_objective="greedy"`. Added
rotations shallow-diagonalize the current remaining one-body matrix
sequentially. Their captured diagonal coefficients seed the extra settings;
the existing rotation bank fills the remainder, followed by an affine
equality correction and materialization checks. The benchmark procedure does
not refine the added angles against the complete projector dictionary.

In the experiment files, a rank or `K_source_terms` counts fitted
density–density factors: this is the manuscript's `K_dd`. The complete
decomposition has `K = K_dd + L_Kdd` terms, including added one-body settings.
The number of active settings can be smaller when constant settings are
removed. GPD's `K` instead counts its sequential retained fragments.

The dense `mpqc_measurement.SRDDConfig` API and `portable/` interface use the
small-system equal-stratum/full-row-rank profile. Their defaults, seeds, and
optimizer controls are not the complete published configuration for every
molecule. Use the corresponding experiment drivers and saved audit metadata
for benchmark reproduction; the BeH2/N2 tensor drivers preserve their own
positive-only frontier, one-added-rotation policy, and calibration fit.
