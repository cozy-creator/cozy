"""Stages one Runtime update's wheels on the worker for the fixed Creator transport.

A published wheel is fetched here from the release index and checked against its
recorded digest and length, so it never crosses the maintenance SSH connection. A
local build is uploaded by the client; this reports how much of it has arrived.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

STAGED = Path("/var/lib/cozy/dev/staged")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            hasher.update(chunk)
    return hasher.hexdigest()


def fetch(url: str, path: Path, sha256: str, length: int) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "files.pythonhosted.org" or parsed.query or parsed.fragment:
        raise ValueError("Runtime update wheel URL is outside the first-party release index")
    partial = path.with_name(path.name + ".download")
    hasher, size = hashlib.sha256(), 0
    try:
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
            if response.geturl() != url:
                raise ValueError("Runtime update wheel URL unexpectedly redirected")
            while chunk := response.read(1 << 20):
                size += len(chunk)
                if size > length:
                    raise ValueError(f"{path.name} exceeds its published length")
                hasher.update(chunk)
                out.write(chunk)
        if size != length or hasher.hexdigest() != sha256:
            raise ValueError(f"{path.name} failed published digest verification on the worker")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def main() -> dict[str, object]:
    job = json.loads(sys.argv[1])
    directory = Path(job["directory"])
    if directory.parent != STAGED or not re.fullmatch(r"[a-f0-9]{32}", directory.name):
        raise ValueError("Invalid Runtime update staging directory")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    staged: dict[str, object] = {}
    for row in job["wheels"]:
        name, sha256, length, url = row["file"], row["sha256"], row["length"], row.get("url", "")
        if not re.fullmatch(r"[A-Za-z0-9_.+-]+\.whl", name) or not re.fullmatch(r"[a-f0-9]{64}", sha256):
            raise ValueError("Invalid Runtime update wheel identity")
        path = directory / name
        size = path.stat().st_size if path.is_file() else -1
        complete = size == length and digest(path) == sha256
        if not complete and size >= length:
            # Full-length bytes that fail the digest are never resumed onto.
            path.unlink()
            size = -1
        if not complete and url:
            fetch(url, path, sha256, length)
            size, complete = length, True
        staged[name] = {"size": max(size, 0), "exists": size >= 0, "complete": complete}
    return {"staged": staged}


if __name__ == "__main__":
    print(json.dumps(main()))
