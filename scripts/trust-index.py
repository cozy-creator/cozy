#!/usr/bin/env python3
"""proto-061: trust checks answer from one per-process catalog load; bad bytes still refuse."""

import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import tensorfs
from tensorfs.errors import Refusal

LAND = """
import io, sys, tensorfs
store = tensorfs.Store.open(sys.argv[1])
for index in range(int(sys.argv[2])):
    body = index.to_bytes(8, "little") * 1024
    store.put_reader(io.BytesIO(body), tensorfs.object_id(body), len(body))
"""


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="tensorfs-trust-index-") as directory:
        root = Path(directory) / "store"
        tensorfs.Store.init(root)
        count = 256
        # Landed by another process: this one has never seen a verification row.
        subprocess.run([sys.executable, "-c", LAND, str(root), str(count)], check=True)
        bodies = [index.to_bytes(8, "little") * 1024 for index in range(count)]
        digests = [tensorfs.object_id(body) for body in bodies]
        store = tensorfs.Store.open(root)
        assert store.verify(digests[0]) == {"verdict": "verified", "rehashed": False}

        # The index is loaded. With the catalog unopenable, a per-object catalog open
        # would turn every check below into a rehash that cannot record, and refuse.
        catalog = root / "tensorfs.sqlite"
        os.chmod(catalog, 0)
        try:
            for _ in range(3):
                for digest in digests:
                    assert store.verify(digest) == {"verdict": "verified", "rehashed": False}
        finally:
            os.chmod(catalog, 0o644)

        # Changed bytes break the record's binding, are rehashed, and are removed.
        hexdigest = digests[1].removeprefix("sha256:")
        blob = root / "blobs" / hexdigest[:2] / hexdigest[2:4] / hexdigest
        os.chmod(blob, 0o644)
        blob.write_bytes(b"\xa5" * len(bodies[1]))
        assert store.verify(digests[1])["verdict"] == "corrupt_removed"
        assert not blob.exists()

        # Landing hashes: bytes that are not the declared object are never installed.
        wrong = b"\x07" * 5000
        declared = tensorfs.object_id(b"\x08" * 5000)
        try:
            store.put_reader(io.BytesIO(wrong), declared, len(wrong))
        except Refusal as error:
            assert error.code == "OBJECT_ID_MISMATCH", error
        else:
            raise AssertionError("mismatched bytes were admitted")
        assert not store.contains(declared.removeprefix("sha256:"))
        print(f"trust-index: {count} objects x3 verified with the catalog unopenable; "
              "changed and mismatched bytes refused")


if __name__ == "__main__":
    main()
