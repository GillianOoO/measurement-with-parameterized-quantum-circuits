"""Append the accepted 50+50 random-Hamiltonian experiment to the data release.

This maintainer command copies scientific files losslessly into independent
ZIP shards. It never reruns an experiment or replaces an existing dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

BUNDLE = Path(__file__).resolve().parents[1]
PREFIX = "data/random_ensemble_50"
OMITTED_COMPONENTS = {
    "calibration_approximation_components.csv",
    "calibration_sampling_components.csv",
}
ALLOWED_SUFFIXES = {".npy", ".npz", ".csv", ".json", ".txt"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_files(run: Path) -> dict[str, Path]:
    metadata = json.loads((run / "inputs/manifest.json").read_text(encoding="utf-8"))
    if metadata.get("instances_per_regime") != 50 or metadata.get("case_count") != 100:
        raise ValueError("Expected the completed 50-instance ensemble in each regime")
    selected = {}

    def add(file: Path) -> None:
        if not file.is_file() or file.suffix not in ALLOWED_SUFFIXES:
            return
        if not file.resolve().is_relative_to(run):
            raise ValueError(f"Input escapes production run: {file}")
        relative = file.relative_to(run)
        if "__pycache__" in relative.parts or file.name in OMITTED_COMPONENTS:
            return
        selected[f"{PREFIX}/{relative.as_posix()}"] = file

    for folder in ("inputs", "results"):
        for file in sorted((run / folder).rglob("*")):
            add(file)
    for regime in ("sparse", "dense"):
        for index in range(1, 51):
            case = run / "cases" / f"{regime}4_seed0_{index}"
            complete = case / "complete.json"
            if not complete.is_file():
                raise FileNotFoundError(complete)
            record = json.loads(complete.read_text(encoding="utf-8"))
            if record.get("binding", {}).get("repeats") != 50:
                raise ValueError(f"Unexpected measurement repetition count: {case.name}")
            add(complete)
            for folder in ("gpd_source", "gpd_selected", "pauli"):
                if not (case / folder).is_dir():
                    raise FileNotFoundError(case / folder)
                for file in sorted((case / folder).rglob("*")):
                    add(file)
    for name in ("timing_decision.json", "legacy_prefix_reproduction.json"):
        add(run / name)
    required = ["results/replicates.csv", "results/instance_results.csv",
                "results/ensemble_summary.csv", "results/paired_input_audit.csv",
                "inputs/common_input_state.npy"]
    for relative in required:
        if f"{PREFIX}/{relative}" not in selected:
            raise FileNotFoundError(run / relative)
    return dict(sorted(selected.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    files = selected_files(args.run_root.resolve())
    destination = BUNDLE / "datasets"
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = {}
    for relative, file in files.items():
        prior = manifest["files"].get(relative)
        if prior:
            if prior["sha256"] != sha(file) or prior["size"] != file.stat().st_size:
                raise ValueError(f"Preserving different released scientific data: {relative}")
        else:
            missing[relative] = file
    if args.check_only:
        if missing:
            raise ValueError(f"{len(missing)} ensemble files are not yet packaged")
        print(f"Verified {len(files)} released ensemble files against production records")
        return
    if not missing:
        print(f"All {len(files)} ensemble files are already present and unchanged")
        return
    shards, current, size = [], [], 0
    for relative, file in missing.items():
        file_size = file.stat().st_size
        if current and size + file_size > 64 * 1024**2:
            shards.append(current)
            current, size = [], 0
        current.append((relative, file))
        size += file_size
    if current:
        shards.append(current)
    indices = [int(match.group(1)) for name in manifest["archives"]
               if (match := re.fullmatch(r"numerical_data_(\d+)\.zip", name))]
    start = max(indices, default=0) + 1
    for index, shard in enumerate(shards, start):
        name = f"numerical_data_{index:02d}.zip"
        target = destination / name
        with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for relative, file in shard:
                archive.write(file, relative)
                manifest["files"][relative] = {
                    "sha256": sha(file), "size": file.stat().st_size, "archive": name,
                }
        if target.stat().st_size >= 95 * 1024**2:
            raise ValueError(f"Archive exceeds the release size limit: {name}")
        manifest["archives"][name] = {"sha256": sha(target), "size": target.stat().st_size}
        print(f"{name}: {len(shard)} files, {target.stat().st_size / 1024**2:.2f} MiB", flush=True)
    manifest.setdefault("supplements", {})["random_ensemble_50"] = {
        "prefix": PREFIX,
        "instances_per_regime": 50,
        "measurement_repetitions_per_point": 50,
        "production_record_policy": "Original scientific file bytes preserved",
        "omitted": ["logs", "superseded Pauli arithmetic diagnostics", "rendered figures",
                    "progress.json", *sorted(OMITTED_COMPONENTS)],
        "calibration_reconstruction": "Saved probe arrays, source fragments, weights and frontiers are retained",
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    print(f"Packaged {len(missing)} new files; all prior dataset entries retained")


if __name__ == "__main__":
    main()
