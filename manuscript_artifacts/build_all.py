"""Rebuild current figures from code and user-supplied, hash-verified local data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BUNDLE = HERE / "reproduction"


def verify_bundle(*, check_inputs=True):
    manifest = json.loads((BUNDLE / "bundle_manifest.json").read_text(encoding="utf-8"))
    files = dict(manifest["files"])
    if check_inputs:
        files.update(json.loads((BUNDLE / manifest["required_inputs"]).read_text())["files"])
    for relative, expected in files.items():
        source = (BUNDLE / relative).resolve()
        if not source.is_relative_to(BUNDLE.resolve()):
            raise ValueError(f"Invalid bundle path: {relative}")
        if not source.is_file():
            raise FileNotFoundError(f"Missing local input {relative}. Run manuscript_artifacts/prepare_inputs.py --archive-root PATH_TO_paper_reproducibility first.")
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Bundle hash mismatch: {relative}")
    return len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE.parent / "build/manuscript")
    parser.add_argument("--figures", nargs="+", choices=("1", "2", "3", "4", "5", "s1"))
    parser.add_argument("--skip-reference-check", action="store_true",
                        help="Allow renderer/font differences; the result is marked RENDERED_UNCHECKED.")
    args = parser.parse_args()
    print(f"Verified code and local inputs: {verify_bundle()} files", flush=True)
    command = [sys.executable, str(BUNDLE / "reproduce_manuscript_figures.py"),
               "--output", str(args.output.resolve())]
    if args.figures:
        command.extend(["--figures", *args.figures])
    if args.skip_reference_check:
        command.append("--skip-reference-check")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
