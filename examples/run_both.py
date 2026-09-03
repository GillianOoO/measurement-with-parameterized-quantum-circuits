"""Small API example. Reduce GPD controls here only for a quick local run."""

from pathlib import Path

from mpqc_measurement import GPDConfig, SRDDConfig, run_gpd, run_srdd


HERE = Path(__file__).resolve().parent
INPUT = HERE / "h2_pauli.json"
OUTPUT = HERE.parent / "results"

gpd = run_gpd(
    INPUT,
    shots=100,
    config=GPDConfig(
        paper_depth=1,
        max_terms=2,
        steps=10,
        calibrate=False,
    ),
    output=OUTPUT / "gpd",
)

srdd = run_srdd(
    INPUT,
    shots=100,
    config=SRDDConfig(calibrate=False),
    output=OUTPUT / "srdd",
)

for result in (gpd, srdd):
    print(
        result.method,
        "settings=", len(result.fragments),
        "bound=", result.error.state_independent_rmse_bound,
    )
