"""Verify and unpack the bundled numerical data, without network or local archives."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

BUNDLE = Path(__file__).resolve().parent / "reproduction"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def materialize(*, verify_only=False):
    root = BUNDLE.resolve()
    manifest = json.loads((root / "datasets/manifest.json").read_text())
    grouped = {}
    for relative, record in manifest["files"].items():
        parts = PurePosixPath(relative)
        target = (root / relative).resolve()
        if (parts.is_absolute() or ".." in parts.parts or "\\" in relative
                or not relative.startswith("data/") or not target.is_relative_to(root / "data")):
            raise ValueError(f"Unsafe dataset path: {relative}")
        if target.exists():
            if not target.is_file() or sha(target) != record["sha256"]:
                raise ValueError(f"Preserving a different existing file: {target}")
        elif verify_only:
            raise FileNotFoundError(f"Run materialize_data.py first: {relative}")
        else:
            grouped.setdefault(record["archive"], []).append((relative, target, record))
    for archive, record in manifest["archives"].items():
        if Path(archive).name != archive:
            raise ValueError(f"Unsafe archive path: {archive}")
        file = root / "datasets" / archive
        if sha(file) != record["sha256"]:
            raise ValueError(f"Archive hash mismatch: {archive}")
        if archive not in grouped:
            continue
        with zipfile.ZipFile(file) as z:
            for relative, target, expected in grouped[archive]:
                info = z.getinfo(relative)
                if info.file_size != expected["size"]:
                    raise ValueError(f"Size mismatch: {relative}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temp:
                    staging = Path(temp.name)
                    try:
                        with z.open(info) as src:
                            shutil.copyfileobj(src, temp)
                    except BaseException:
                        temp.close()
                        staging.unlink(missing_ok=True)
                        raise
                if sha(staging) != expected["sha256"]:
                    staging.unlink()
                    raise ValueError(f"Extracted hash mismatch: {relative}")
                staging.replace(target)
    return len(manifest["files"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    print(f"Verified {materialize(verify_only=args.verify_only)} numerical files")
