"""Compare all replayed Figure 3 Born outcomes with the published fixed-seed data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ARCHIVE = Path(__file__).resolve().parents[2]


def verify(replay: Path, data_root: Path):
    replay, data_root = replay.resolve(), data_root.resolve()
    frozen = data_root / "shared_processed/figure3/empirical50"
    if replay == frozen:
        raise ValueError("Replay must be a separate output directory, not the published reference")
    checks = []
    for directory, counts, fields in (
        ("settings", (("BeH2", 21), ("N2", 41)),
         ("alpha_outcomes", "beta_outcomes", "contributions", "seed", "count")),
        ("fc_groups", (("BeH2", 36), ("N2", 111)),
         ("outcomes", "contributions", "seed", "count", "cnot_count",
          "output_z_masks", "pooled_score_coefficients")),
    ):
        for molecule, count in counts:
            for index in range(count):
                relative = Path(directory) / f"{molecule}_{index:03d}.npz"
                source, actual = frozen / relative, replay / relative
                with np.load(source, allow_pickle=False) as expected, np.load(actual, allow_pickle=False) as result:
                    different = [key for key in fields if not np.array_equal(expected[key], result[key])]
                checks.append(dict(path=relative.as_posix(), identical_arrays=not different,
                                   differing_fields=different,
                                   reference_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                                   replay_sha256=hashlib.sha256(actual.read_bytes()).hexdigest()))
    result = dict(status="PASS" if all(row["identical_arrays"] for row in checks) else "DIFFERENT",
                  scope="Fixed-seed noisy outcome and contribution arrays; ZIP container bytes may differ",
                  setting_group_files=len(checks), identical_files=sum(row["identical_arrays"] for row in checks),
                  checks=checks)
    (replay / "published_array_comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "setting_group_files", "identical_files")}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ARCHIVE / "data")
    args = parser.parse_args()
    raise SystemExit(0 if verify(args.replay, args.data_root)["status"] == "PASS" else 1)
