"""Rebuild the current main and SI numerical tables from released records."""
import argparse
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE.parent / "build/manuscript/tables")
    args = parser.parse_args()
    subprocess.run([sys.executable,
                    str(HERE / "reproduction/code/analysis/rebuild_manuscript_tables.py"),
                    "--output", str(args.output.resolve())], check=True)
