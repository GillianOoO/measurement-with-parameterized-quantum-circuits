"""Paths shared by the released SRDD drivers; independent of checkout name."""

import os
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("MEASUREMENT_DATA_ROOT", BUNDLE / "data")).resolve()
METHODS = BUNDLE / "code" / "methods"
RUNS = BUNDLE / "runs"


def molecular_result(root: Path, molecule: str, role: str) -> Path:
    """Accept both the released per-molecule tree and a fresh runner output."""
    names = {
        ("H4", "ranks"): "srdd_fixed_rank_fits",
        ("H4", "selected"): "srdd_fixed_selected",
        ("H6", "ranks"): "srdd_rank_fits",
        ("H6", "selected"): "srdd_selected",
    }
    released = root / molecule / "results" / names[(molecule, role)]
    return released if released.is_dir() else root / molecule
