"""Import required plotting inputs from a user-provided local archive; no network access."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

BUNDLE = Path(__file__).resolve().parent / "reproduction"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", required=True, type=Path,
                        help="Local paper_reproducibility directory containing the numerical inputs and figure references")
    args = parser.parse_args()
    source_root = args.archive_root.resolve()
    required = json.loads((BUNDLE / "required_inputs.json").read_text())["files"]
    for relative, expected in required.items():
        source = (source_root / relative).resolve()
        target = (BUNDLE / relative).resolve()
        if not source.is_relative_to(source_root) or not target.is_relative_to(BUNDLE.resolve()):
            raise ValueError(f"Invalid input path: {relative}")
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source input hash mismatch: {relative}")
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Preserving a different existing local input: {target}")
    for relative in required:
        source, target = source_root / relative, BUNDLE / relative
        if source.resolve() != target.resolve():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    print(f"Imported {len(required)} verified local inputs; they remain gitignored")


if __name__ == "__main__":
    main()
