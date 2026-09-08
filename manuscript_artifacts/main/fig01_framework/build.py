#!/usr/bin/env python3
"""Copy the current reviewed Figure 1; the editable PPTX is distributed separately."""

import argparse
from pathlib import Path
import shutil
import sys

ARTIFACTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ARTIFACTS))
from build_all import verify_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify_bundle()
    source = ARTIFACTS / "reproduction/figures/published/main_sketch_revise.pdf"
    if args.output.resolve() == source.resolve():
        parser.error("Use an output path other than the frozen source PDF")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, args.output)


if __name__ == "__main__":
    main()
