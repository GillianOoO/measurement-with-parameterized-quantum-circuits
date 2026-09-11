"""Maintainer command: curate lossless data archives from an existing local dataset.

Only numerical inputs/results and their metadata are selected. Private paper,
response, revision and QA directories are never traversed into the release.
The original data files are neither rewritten nor removed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

BUNDLE = Path(__file__).resolve().parents[1]
EXTENSIONS = {".npy", ".npz", ".csv", ".json", ".xyz"}
EXCLUDED_DIRS = {"before", "build", "qa", "figures", "logs", "__pycache__"}
EXCLUDED_NAMES = {"text_revisions.json", "revision_manifest.json"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive-root", required=True, type=Path)
    p.add_argument("--supplement", action="store_true",
                   help="Add hash-distinct missing data from another curated source; never replace existing files")
    args = p.parse_args()
    source = args.archive_root.resolve()
    dest = BUNDLE / "datasets"
    dest.mkdir(parents=True, exist_ok=True)
    if list(dest.glob("*.zip")) and not args.supplement:
        raise FileExistsError("Preserving existing archives; choose an empty release directory")
    prior = json.loads((dest / "manifest.json").read_text()) if args.supplement else None
    required = json.loads((BUNDLE / "required_inputs.json").read_text())["files"]
    selected = {}
    for file in sorted((source / "data").rglob("*")):
        if not file.is_file():
            continue
        relative = file.relative_to(source)
        if EXCLUDED_DIRS.intersection(relative.parts) or file.name in EXCLUDED_NAMES:
            continue
        numeric_text = file.suffix.lower() == ".txt" and "inputs" in relative.parts
        if file.suffix.lower() in EXTENSIONS or numeric_text or relative.as_posix() in required:
            if prior and relative.as_posix() in prior["files"]:
                if sha(file) != prior["files"][relative.as_posix()]["sha256"]:
                    raise ValueError(f"Refusing changed scientific data: {relative}")
                continue
            selected[relative.as_posix()] = file
    missing = [r for r in required if r.startswith("data/") and r not in selected]
    if missing and not args.supplement:
        raise ValueError(f"Required files excluded from numerical release: {missing}")
    # Independent shards keep each GitHub object small. A single large dense
    # matrix stays intact inside one lossless compressed archive.
    shards, current, size = [], [], 0
    for relative, file in selected.items():
        nbytes = file.stat().st_size
        if current and size + nbytes > 64 * 1024**2:
            shards.append(current)
            current, size = [], 0
        current.append((relative, file))
        size += nbytes
    if current:
        shards.append(current)
    manifest = prior or {"schema": 1, "compression": "ZIP DEFLATE; original bytes preserved",
                "excluded": ["manuscript/response drafts", "revision notes", "logs", "QA screenshots"],
                "archives": {}, "files": {}}
    for i, shard in enumerate(shards, len(manifest["archives"]) + 1):
        name = f"numerical_data_{i:02d}.zip"
        target = dest / name
        with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for relative, file in shard:
                z.write(file, relative)
                manifest["files"][relative] = {"sha256": sha(file), "size": file.stat().st_size,
                                              "archive": name}
        if target.stat().st_size >= 95 * 1024**2:
            raise ValueError(f"Archive too large for release: {name}")
        manifest["archives"][name] = {"sha256": sha(target), "size": target.stat().st_size}
        print(f"{name}: {len(shard)} files; {target.stat().st_size / 1024**2:.2f} MiB", flush=True)
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    # Only the explicit public plot reference list is copied, never paper drafts.
    for relative, expected in required.items():
        if relative.startswith("data/") or args.supplement:
            continue
        file = source / relative
        if sha(file) != expected:
            raise ValueError(f"Plot reference differs: {relative}")
        target = BUNDLE / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)
    print(f"Packaged {len(selected)} numerical files in {len(shards)} archives")


if __name__ == "__main__":
    main()
