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
