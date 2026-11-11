"""Build current 50-instance Main Figure 5 using the canonical runner."""
from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(here.parents[1] / "build.py"), str(here), *sys.argv[1:]], check=True)
