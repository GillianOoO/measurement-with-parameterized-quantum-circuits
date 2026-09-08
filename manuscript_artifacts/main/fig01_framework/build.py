#!/usr/bin/env python3
"""Build the source-verified first-slide framework PDF and PNG."""

import argparse
from pathlib import Path
import sys

ARTIFACTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ARTIFACTS))
sys.path.insert(0, str(ARTIFACTS / "reproduction/code/plotting"))
from export_framework import copy_reviewed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".pdf":
        parser.error("Output must end in .pdf")
    copy_reviewed(args.output, args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
