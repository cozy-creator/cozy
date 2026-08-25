#!/usr/bin/env python3
"""Red arms for cl-009: build each hostile input FOR REAL, run the real `cozy`
binary against it, observe the typed refusal, then remove the artifact.

Not a test suite (decisions.md #160) — an orchestration script that drives live
runs of the shipped binary and prints what it observed.

    scripts/redarm.py --cozy ./cozy --tree <endpoint tree> [--only <name>]
"""
import argparse
import hashlib
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

EMPTY_SHA = hashlib.sha256(b"").hexdigest()


def decl(files, endpoint="cozy-creator/demo", version="1.0.0"):
    return json.dumps({"endpoint": endpoint, "version": version, "files": files}).encode()


def entry(tar, name, data, mode=0o644, typeflag=tarfile.REGTYPE, linkname=""):
    info = tarfile.TarInfo(name)
    info.type, info.mode, info.mtime = typeflag, mode, int(time.time())
    info.linkname = linkname
    if typeflag in (tarfile.REGTYPE, tarfile.AREGTYPE):
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    else:
        info.size = 0
        tar.addfile(info)


class Zeros:
    """A lazily generated stream — a gzip bomb without a bomb on disk."""

    def __init__(self, n):
        self.left = n

    def read(self, size=-1):
        if size < 0 or size > self.left:
            size = self.left
        self.left -= size
        return b"\0" * size


def hostile(path, build):
    with tarfile.open(path, "w:gz") as tar:
        build(tar)
    return path


# ---- arms: each returns (archive_path, extra_cli_args) ----------------------

def arm_traversal(d):
    def b(tar):
        entry(tar, "release.json", decl([{"path": "../../escaped.py", "sha256": EMPTY_SHA, "size": 0}]))
        entry(tar, "../../escaped.py", b"")
    return hostile(d / "a.tar.gz", b), []


def arm_absolute(d):
    def b(tar):
        entry(tar, "release.json", decl([{"path": "/etc/cozy-owned", "sha256": EMPTY_SHA, "size": 0}]))
        entry(tar, "/etc/cozy-owned", b"")
    return hostile(d / "a.tar.gz", b), []


def arm_symlink(d):
    def b(tar):
        entry(tar, "release.json", decl([{"path": "pyproject.toml", "sha256": EMPTY_SHA, "size": 0}]))
        entry(tar, "pyproject.toml", b"", typeflag=tarfile.SYMTYPE, linkname="/etc/passwd")
    return hostile(d / "a.tar.gz", b), []


def arm_device(d):
    def b(tar):
        entry(tar, "release.json", decl([{"path": "pipe", "sha256": EMPTY_SHA, "size": 0}]))
        entry(tar, "pipe", b"", typeflag=tarfile.FIFOTYPE)
    return hostile(d / "a.tar.gz", b), []


def arm_duplicate(d):
    def b(tar):
        entry(tar, "release.json", decl([{"path": "a.py", "sha256": EMPTY_SHA, "size": 0}]))
        entry(tar, "a.py", b"")
        entry(tar, "./a.py", b"")
    return hostile(d / "a.tar.gz", b), []


def arm_case_collision(d):
    def b(tar):
        entry(tar, "release.json", decl([
            {"path": "Model.py", "sha256": EMPTY_SHA, "size": 0},
            {"path": "model.py", "sha256": EMPTY_SHA, "size": 0}]))
        entry(tar, "Model.py", b"")
        entry(tar, "model.py", b"")
    return hostile(d / "a.tar.gz", b), []


def arm_undeclared(d):
    def b(tar):
        entry(tar, "release.json", decl([{"path": "a.py", "sha256": EMPTY_SHA, "size": 0}]))
        entry(tar, "a.py", b"")
        entry(tar, "sitecustomize.py", b"import os")
    return hostile(d / "a.tar.gz", b), []


def arm_no_declaration(d):
    def b(tar):
        entry(tar, "a.py", b"")
        entry(tar, "release.json", decl([{"path": "a.py", "sha256": EMPTY_SHA, "size": 0}]))
    return hostile(d / "a.tar.gz", b), []


def arm_path_too_long(d):
    long = "x" * 300 + ".py"
    def b(tar):
        entry(tar, "release.json", decl([{"path": long, "sha256": EMPTY_SHA, "size": 0}]))
        entry(tar, long, b"")
    return hostile(d / "a.tar.gz", b), []


def arm_too_many_files(d):
    names = [f"f{i}.py" for i in range(20001)]
    def b(tar):
        entry(tar, "release.json", decl([{"path": n, "sha256": EMPTY_SHA, "size": 0} for n in names]))
        for n in names:
            entry(tar, n, b"")
    return hostile(d / "a.tar.gz", b), []


def arm_expanded_cap(d):
    size = (1 << 30) + (1 << 20)
    def b(tar):
        entry(tar, "release.json", decl([{"path": "big.bin", "sha256": EMPTY_SHA, "size": size}]))
        info = tarfile.TarInfo("big.bin")
        info.size, info.mode, info.mtime = size, 0o644, int(time.time())
        tar.addfile(info, Zeros(size))
    return hostile(d / "a.tar.gz", b), []


def arm_compressed_cap(d):
    # Incompressible bytes, so the ARCHIVE itself passes the compressed cap.
    size = 300 << 20
    blob = d / "blob.bin"
    with open(blob, "wb") as f:
        for _ in range(300):
            f.write(os.urandom(1 << 20))
    path = d / "a.tar.gz"
    with tarfile.open(path, "w:gz", compresslevel=1) as tar:
        entry(tar, "release.json", decl([{"path": "blob.bin", "sha256": EMPTY_SHA, "size": size}]))
        tar.add(blob, arcname="blob.bin", recursive=False)
    blob.unlink()
    return path, []


def arm_file_digest(d):
    def b(tar):
        entry(tar, "release.json", decl([{"path": "a.py", "sha256": EMPTY_SHA, "size": 0}]))
        entry(tar, "a.py", b"import os  # not what was declared")
    return hostile(d / "a.tar.gz", b), []


# ---- arms over the real release archive ------------------------------------

def real_archive(d, tree, version="1.0.0", endpoint="cozy-creator/demo"):
    out = d / f"real-{version}.tar.gz"
    here = pathlib.Path(__file__).parent
    r = subprocess.run([sys.executable, str(here / "pack.py"), str(tree), endpoint, version, str(out)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("pack failed: " + r.stderr)
    digest = [l.split()[-1] for l in r.stdout.splitlines() if l.startswith("digest:")][0]
    return out, digest


ARMS = [
    ("traversal", arm_traversal, 3, "archive_traversal"),
    ("absolute-path", arm_absolute, 3, "archive_absolute_path"),
    ("symlink", arm_symlink, 3, "archive_link"),
    ("device-entry", arm_device, 3, "archive_special_entry"),
    ("duplicate-name", arm_duplicate, 3, "archive_duplicate_entry"),
    ("case-fold-collision", arm_case_collision, 3, "archive_case_collision"),
    ("undeclared-entry", arm_undeclared, 3, "archive_undeclared_entry"),
    ("declaration-not-first", arm_no_declaration, 3, "archive_missing_declaration"),
    ("path-too-long", arm_path_too_long, 3, "archive_path_too_long"),
    ("too-many-files", arm_too_many_files, 3, "archive_too_many_files"),
    ("expanded-cap", arm_expanded_cap, 3, "archive_expanded_cap"),
    ("compressed-cap", arm_compressed_cap, 3, "archive_compressed_cap"),
    ("file-digest-mismatch", arm_file_digest, 3, "archive_file_digest_mismatch"),
]


def run(cozy, home, args):
    env = dict(os.environ, COZY_HOME=str(home), NO_COLOR="1")
    started = time.time()
    r = subprocess.run([str(cozy)] + args, capture_output=True, text=True, env=env)
    return r, time.time() - started


def name_of(r):
    for line in (r.stdout + r.stderr).splitlines():
        if line.startswith("error("):
            return line[len("error("):line.index(")")]
    return "<none>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cozy", required=True)
    ap.add_argument("--tree", required=True, help="a real endpoint tree (the legitimate subject)")
    ap.add_argument("--only")
    a = ap.parse_args()
    cozy, tree = pathlib.Path(a.cozy).resolve(), pathlib.Path(a.tree).resolve()

    work = pathlib.Path(tempfile.mkdtemp(prefix="cozy-redarm-"))
    home = work / "home"
    failures, ran = [], 0
    try:
        real, digest = real_archive(work, tree)

        arms = list(ARMS)
        # Source-verification and target arms ride the REAL archive.
        arms += [
            ("source-digest-mismatch", lambda d: (real, ["--digest", "sha256:" + "0" * 64]), 3, "source_digest_mismatch"),
            ("unsigned-by-default", lambda d: (real, []), 3, "source_unverified"),
            ("wrong-major-requested", lambda d: (real, ["--digest", digest]), 3, "release_mismatch"),
        ]

        for name, build, want_code, want_name in arms:
            if a.only and a.only != name:
                continue
            ran += 1
            d = work / name
            d.mkdir(parents=True, exist_ok=True)
            archive, extra = build(d)
            ref = "cozy-creator/demo@v9" if name == "wrong-major-requested" else "cozy-creator/demo"
            args = ["install", ref, "--from", str(archive)] + extra
            if name not in ("unsigned-by-default", "source-digest-mismatch", "wrong-major-requested"):
                args += ["--allow-unsigned"]  # isolate the arm under test
            r, took = run(cozy, home, args)
            got = name_of(r)
            ok = r.returncode == want_code and got == want_name
            print(f"[{'PASS' if ok else 'FAIL'}] {name:24s} exit={r.returncode} error({got}) {took:.2f}s")
            if not ok:
                failures.append(name)
                print("        want exit=%d error(%s)" % (want_code, want_name))
                print("        " + (r.stdout + r.stderr).strip().replace("\n", "\n        ")[:900])
            shutil.rmtree(d, ignore_errors=True)

        print(f"\n{ran - len(failures)}/{ran} red arms observed refusing as designed")
        return 1 if failures else 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
