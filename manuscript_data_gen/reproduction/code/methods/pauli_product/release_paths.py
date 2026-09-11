"""Portable locations for the archived manuscript computation modules."""
from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path, PurePosixPath

BUNDLE_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("MEASUREMENT_DATA_ROOT", BUNDLE_ROOT / "data")).resolve()
OUTPUT_ROOT = Path(os.environ.get("MEASUREMENT_OUTPUT_ROOT", BUNDLE_ROOT / "regenerated")).resolve()


@lru_cache(maxsize=1)
def _data_index() -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in DATA_ROOT.rglob("*"):
        if path.is_file():
            index.setdefault(path.name, []).append(path)
    return index


def archived_path(value: str | Path, expected_sha256: str | None = None) -> Path:
    """Resolve a recorded input after relocation, retaining content checks.

    The original manifests remain unmodified provenance. A unique basename is
    sufficient; ambiguous candidates require either the expected file digest or
    a unique longest matching path suffix. Missing/ambiguous inputs fail closed.
    """
    raw = str(value).replace("\\", "/")
    given = Path(value)
    candidates = list(_data_index().get(PurePosixPath(raw).name, []))
    if given.is_file() and given not in candidates:
        candidates.append(given)
    if expected_sha256 is not None:
        candidates = [p for p in candidates if hashlib.sha256(p.read_bytes()).hexdigest() == expected_sha256]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        old_parts = PurePosixPath(raw).parts
        def score(path: Path) -> int:
            result = 0
            for left, right in zip(reversed(old_parts), reversed(path.parts)):
                if left != right:
                    break
                result += 1
            return result
        maximum = max(map(score, candidates))
        winners = [p for p in candidates if score(p) == maximum]
        if len(winners) == 1:
            return winners[0]
        # Identical duplicated inputs are interchangeable, unlike different data.
        if len({hashlib.sha256(p.read_bytes()).hexdigest() for p in winners}) == 1:
            return sorted(winners)[0]
    raise FileNotFoundError(f"Missing or ambiguous archived input {raw!r} under {DATA_ROOT}; materialize the release data first")
