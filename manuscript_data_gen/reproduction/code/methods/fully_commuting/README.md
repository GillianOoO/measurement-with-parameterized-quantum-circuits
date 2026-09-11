# Fully commuting baseline

The BeH2/N2 baseline is overlapping fully commuting measurement with
iterative measurement allocation (FC-IMA). The code implements extended
sorted insertion, Clifford diagonalization, the pooled Pauli estimator,
integer allocation and exact noise moments for local depolarization after
each compiled CNOT.

Run from `manuscript_data_gen/reproduction` after materializing `data/`.
Dependencies are NumPy, SciPy and Qiskit. Inputs are read from
`data/BeH2/inputs` and `data/N2/inputs`; `MEASUREMENT_DATA_ROOT` optionally
overrides the data root. Recomputed files default to `regenerated/`.

```sh
python -m unittest discover -s code/methods/fully_commuting -p "test_*.py"
python code/methods/fully_commuting/traditional_cm_depth.py --molecule all --output-dir regenerated/fully_commuting/depth
python code/methods/fully_commuting/traditional_cm_error_eval.py --cases BeH2 N2 --budgets 1300 1600 2000 2400 3000 --repeats 50 --output regenerated/fully_commuting/error_eval/results
python code/methods/fully_commuting/traditional_cm_noise_eval.py
```

The noise runner reads the archived frozen T=3000 allocation from
`data/fully_commuting/BeH2_N2_FC_IMA/error_eval/results`, and writes to
`regenerated/fully_commuting/noise_eval/results`. Its default is the ten-point
inclusive p=0,...,0.003 grid.

All 24 focused exactness tests pass in Python 3.12 with NumPy 2.5.3,
SciPy 1.18.1 and Qiskit 2.5.2. A BeH2 synthesis smoke test also passes with
`--molecule BeH2 --max-groups 1`; it is explicitly marked partial.
The full production FC error and noise jobs were not rerun in that smoke test.
