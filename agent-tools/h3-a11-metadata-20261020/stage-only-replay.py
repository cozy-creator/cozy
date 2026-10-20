#!/usr/bin/env python3
"""Hosted-only H3 metadata stage: existing bounded loopback replay, no collector."""
import argparse
import hashlib
import http.server
import json
from pathlib import Path
import re
import subprocess
import threading

CAP = 32 << 20
HEX = re.compile(r"[0-9a-f]{64}\Z")
BUNDLE_SHA256 = "891aeb30592b74beedba96583b7bd130b6730b07fdba8977a23f4d0248811dff"


class StageError(Exception):
    pass


def require(ok, message):
    if not ok:
        raise StageError(message)


def digest(body):
    return hashlib.sha256(body).hexdigest()


def dump(path, value):
    with path.open("x", encoding="utf-8") as out:
        json.dump(value, out, indent=2, sort_keys=True)
        out.write("\n")


def load(path):
    require(path.stat().st_size <= CAP, "JSON document exceeds the metadata cap")
    return json.loads(path.read_bytes())


def reference(value):
    require(
        isinstance(value, dict) and set(value) == {"sha256", "length"},
        "malformed object reference",
    )
    require(
        isinstance(value["sha256"], str) and HEX.fullmatch(value["sha256"]),
        "invalid object digest",
    )
    require(
        type(value["length"]) is int and 0 <= value["length"] <= CAP,
        "object exceeds metadata size cap",
    )
    return value


def checked(body, at):
    reference(at)
    require(
        len(body) == at["length"] and digest(body) == at["sha256"],
        "metadata digest or length mismatch",
    )
    return body


def local_object(directory, at):
    path = directory / at["sha256"]
    require(
        not path.is_symlink()
        and path.is_file()
        and path.stat().st_size == at["length"],
        "export lacks an exact regular metadata file",
    )
    return checked(path.read_bytes(), at)


def stage(args):
    bundle_path = args.bundle.resolve()
    bundle = load(bundle_path)
    require(bundle["format"] == "cozy.a11.metadata-bundle/1", "invalid bundle")
    require(
        not args.destination.exists() and not args.destination.is_symlink(),
        "destination must be fresh",
    )
    require(
        not any(
            (bundle_path.parent / name).exists()
            for name in ("replay-requests.json", "stage-receipt.json")
        ),
        "bundle was already staged; preserve its receipts and use a new bundle directory",
    )
    replay = bundle["replay"]
    replay_log = []

    class Replay(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            at = replay.get(self.path)
            if at is None:
                self.send_error(403)
                replay_log.append({"path": self.path, "denied": True})
                return
            data = local_object(bundle_path.parent / "objects", at)
            replay_log.append({"path": self.path, "object": at})
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), Replay) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = subprocess.run(
                [
                    str(args.migrator.resolve()),
                    str(bundle_path),
                    str(args.destination.resolve()),
                    f"http://127.0.0.1:{server.server_port}",
                ],
                check=False,
                capture_output=True,
                timeout=120,
            )
            if result.returncode:
                (bundle_path.parent / "native-error.txt").write_bytes(result.stderr)
                raise StageError(
                    "native migrator failed; see preserved native-error.txt"
                )
            receipt = json.loads(result.stdout)
        finally:
            server.shutdown()
            thread.join(timeout=5)
            dump(bundle_path.parent / "replay-requests.json", replay_log)
    require(
        not any(row.get("denied") for row in replay_log),
        "native migrator requested non-allowlisted upstream metadata",
    )
    receipt.update(
        bundle_sha256=digest(bundle_path.read_bytes()),
        binary_sha256=digest(args.migrator.read_bytes()),
        replay_requests=len(replay_log),
    )
    dump(bundle_path.parent / "stage-receipt.json", receipt)
    print(bundle_path.parent / "stage-receipt.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--migrator", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    require(digest(args.bundle.read_bytes()) == BUNDLE_SHA256, "not the exact H3 bundle")
    stage(args)


if __name__ == "__main__":
    main()
