"""Maintainer-only: record reviewed release source hashes after tests.

This command is not a numerical validation and must not be used to conceal
changed scientific inputs. Their independent, original hashes are retained.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old = json.loads((ROOT / "bundle_manifest.json").read_text())
    selected = {ROOT / p for p in old["files"] if (ROOT / p).is_file()}
    for folder in ("code", "validation"):
        selected.update(p for p in (ROOT / folder).rglob("*")
                        if p.is_file() and p.suffix in {".py", ".md", ".json", ".ps1"}
                        and "__pycache__" not in p.parts
                        and "release_checks" not in p.parts
                        and p.name != "current_bias_validation.json")
    old.update(schema=3,
               scope="Portable manuscript experiment code, public plot assets and lossless numerical data archives",
               numerical_data_manifest="datasets/manifest.json",
               not_included=["private manuscript/reviewer documents", "backup archives", "runtimes and caches"],
               files={p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(selected)})
    old.pop("source_archive", None)
    (ROOT / "bundle_manifest.json").write_text(json.dumps(old, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    required = json.loads((ROOT / "required_inputs.json").read_text())
    required["scope"] = "Plot dependencies supplied by the public numerical archives and figure reference assets"
    (ROOT / "required_inputs.json").write_text(json.dumps(required, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Recorded {len(old['files'])} reviewed source/reference hashes; scientific hashes unchanged")


if __name__ == "__main__":
    main()
