"""Build current SI Figure 1 using the canonical 50-instance runner."""
from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(here.parents[1] / "build.py"), str(here), *sys.argv[1:]], check=True)
