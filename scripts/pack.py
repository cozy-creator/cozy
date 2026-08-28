#!/usr/bin/env python3
"""Pack an endpoint tree into a release archive — the pre-hub stand-in for `cozy endpoint publish`.

cl-012 replaces this with the real publish path (manifest from `git ls-files`,
declare-digests-first, presigned PUTs). Until then this writes the same shape the
installer verifies: release.json FIRST, then exactly the files it declares.

    scripts/pack.py <tree> <org/endpoint> <version> <out.tar.gz>

Prints the archive's sha256 — what `cozy install --digest` checks.
"""
import hashlib
import io
import json
import pathlib
import sys
import tarfile
import time

SKIP = {".git", ".venv", "__pycache__", ".mypy_cache", ".ruff_cache", "node_modules", "dist"}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    if len(sys.argv) != 5:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    tree, endpoint, version, out = sys.argv[1:]
    root = pathlib.Path(tree).resolve()

    files = []
    for p in sorted(root.rglob("*")):
        if any(part in SKIP for part in p.relative_to(root).parts):
            continue
        if not p.is_file() or p.is_symlink():
            continue
        rel = p.relative_to(root).as_posix()
        files.append({"path": rel, "sha256": digest(p), "size": p.stat().st_size})

    decl = json.dumps({"endpoint": endpoint, "version": version, "files": files},
                      indent=2).encode()

    with tarfile.open(out, "w:gz") as tar:
        info = tarfile.TarInfo("release.json")
        info.size, info.mtime, info.mode = len(decl), int(time.time()), 0o644
        tar.addfile(info, io.BytesIO(decl))
        for f in files:
            tar.add(root / f["path"], arcname=f["path"], recursive=False)

    print(f"archive:  {out}")
    print(f"endpoint: {endpoint}  version: {version}  files: {len(files)}")
    print(f"digest:   sha256:{digest(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
