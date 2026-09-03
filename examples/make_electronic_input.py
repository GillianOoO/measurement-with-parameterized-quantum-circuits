"""Create a two-spatial-orbital electronic-integral input for an SRDD smoke run."""

from pathlib import Path

import numpy as np


output = Path(__file__).resolve().parent.parent / "results" / "two_orbital_electronic.npz"
output.parent.mkdir(parents=True, exist_ok=True)
one_body = np.asarray([[-1.0, 0.08], [0.08, 0.45]])
two_body_chemist = np.zeros((2, 2, 2, 2))
two_body_chemist[0, 0, 0, 0] = 0.65
two_body_chemist[1, 1, 1, 1] = 0.52
two_body_chemist[0, 0, 1, 1] = 0.18
two_body_chemist[1, 1, 0, 0] = 0.18
np.savez_compressed(
    output,
    constant=0.1,
    one_body=one_body,
    two_body_chemist=two_body_chemist,
    n_electrons=2,
)
print(output)
