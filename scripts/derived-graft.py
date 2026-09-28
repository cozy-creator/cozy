#!/usr/bin/env python3
"""Source-part inheritance survives loss of the producer process before acknowledgement."""

import copy
import io
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import tensorfs

WORK = "sha256:" + "97" * 32
TRANSACTION = "sha256:" + "98" * 32


def request(store):
    manifests = [store.derived_lookup("sha256:" + suffix * 32)["receipt"]["manifest"] for suffix in ("95", "96")]
    sources = {
        alias: ("sha256:" + manifest["sha256"], manifest["length"])
        for alias, manifest in zip(("body", "bank"), manifests)
    }
    plain = next(digest for alias, digest in tensorfs.seed_digests() if alias == "plain/1")
    targets = {"model": {
        "source": "body", "source_component": "model", "drop": [],
        "add": {"table": {
            "logical_dtype": "f32", "shape": [512], "encoding": plain,
            "parts": {"value": {
                "dtype": "f32", "shape": [512],
                "source": {"source": "bank", "component": "model", "tensor": "weight", "role": "value"},
            }},
        }},
    }}
    return sources, targets, {}, [("model", "weight"), ("model", "table")], 0


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        store = tensorfs.Store.open(sys.argv[2])
        writer = store.begin_derived(TRANSACTION, 1, *request(store), work_fingerprint=WORK)
        writer.commit()
        # The native commit is durable; no host result/receipt acknowledgement follows.
        os.kill(os.getpid(), signal.SIGKILL)
        raise AssertionError("SIGKILL returned")

    with tempfile.TemporaryDirectory(prefix="tensorfs-source-graft-") as root:
        store = tensorfs.Store.init(root)
        plain = next(digest for alias, digest in tensorfs.seed_digests() if alias == "plain/1")
        targets = {"model": {"drop": [], "add": {"weight": {
            "logical_dtype": "f32", "shape": [512], "encoding": plain,
            "parts": {"value": {"dtype": "f32", "shape": [512]}},
        }}}}
        for suffix, byte in (("95", 0x11), ("96", 0x22)):
            writer = store.begin_derived("sha256:" + suffix * 32, 1, {}, targets, {}, [("model", "weight")], 2048, work_fingerprint=WORK)
            writer.add_part("model", "weight", "value", io.BytesIO(bytes([byte]) * 2048))
            writer.commit()
        args = request(store)
        expected = tensorfs.object_id(store.derived_declaration(*args, work_fingerprint=WORK))
        malformed = copy.deepcopy(args)
        malformed[1]["model"]["add"]["table"]["parts"]["value"]["source"]["object"] = {"sha256": "ff" * 32, "length": 2048}
        try:
            store.derived_declaration(*malformed, work_fingerprint=WORK)
        except tensorfs.errors.Refusal as error:
            assert "UNKNOWN_FIELD" in str(error), error
        else:
            raise AssertionError("an author-controlled object escaped the semantic selector")

        child = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", root], check=False)
        assert child.returncode == -signal.SIGKILL, child.returncode
        receipt = tensorfs.Store.open(root).derived_lookup(TRANSACTION)["receipt"]
        assert receipt["declaration_digest"] == expected
        assert receipt["added_objects"] == []
        assert receipt["inherit_observation"] == {"bytes": 4096, "objects": 2, "reads": 0, "hashes": 0}
        for suffix in ("95", "96"):
            store.derived_dispose("sha256:" + suffix * 32)
        manifest = receipt["manifest"]
        store.inspect_derived_source("sha256:" + manifest["sha256"], manifest["length"], ["model"], [])
        print("source graft green: two native sources, zero-byte declaration, strict semantic selector, SIGKILL after commit, receipt readback and independent output custody")


if __name__ == "__main__":
    main()
