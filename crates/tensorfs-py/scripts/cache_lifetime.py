#!/usr/bin/env python3
"""Real process GC must preserve an unread tensor through its last native owner."""

import gc
import json
import os
import subprocess
import sysconfig
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import tensorfs


class CacheLifetime(unittest.TestCase):
    def test_unread_child_owner_survives_then_becomes_collectible(self):
        with TemporaryDirectory(prefix="tensorfs-cache-lifetime-") as temp:
            cache = Path(temp) / "cache"

            def produce(payload):
                with tensorfs.Writer(cache_dir=cache) as writer:
                    writer.component("model", cls="test.Model", config={}).put(
                        "weight", payload, dtype="U8", shape=[len(payload)]
                    )
                checkpoint = writer.commit()
                reference = checkpoint.id
                checkpoint.close()
                return reference

            active = produce(b"unread-active-payload")
            stale = produce(b"unrelated-stale-payload")
            checkpoint = tensorfs.open(active, cache_dir=cache, local_files_only=True)
            component = checkpoint.components["model"]
            tensors = component.tensors
            # No tensor data has been read or opened. The native child owners must retain
            # its CAS name even after the Python Checkpoint and Component are released.
            checkpoint.close()
            del checkpoint, component
            gc.collect()

            def blob(reference):
                digest = reference.removeprefix("sha256:")
                return cache / "blobs" / digest[:2] / digest[2:4] / digest

            def collect_in_other_process():
                # Age only this tiny fixture, beyond the production grace/TTL, without
                # changing policy constants or exhausting the real host filesystem.
                for path in (cache / "blobs").glob("*/*/*"):
                    if path.is_file():
                        os.utime(path, (1, 1))
                for path in (cache / "pins").glob("*.json"):
                    document = json.loads(path.read_text())
                    document["last_use"] = 1
                    path.write_text(json.dumps(document))
                result = subprocess.run(
                    [
                        str(Path(sysconfig.get_path("scripts")) / "tfs"),
                        "--store",
                        str(cache),
                        "--json",
                        "gc",
                        "--older-than",
                        "0s",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                return json.loads(result.stdout)

            self.assertTrue(blob(stale).is_file())
            first = collect_in_other_process()
            self.assertGreater(first["objects"], 0, "the eviction control did no work")
            self.assertFalse(
                blob(stale).exists(), "unrelated closed checkpoint was retained"
            )
            self.assertEqual(tensors.read("weight").tobytes(), b"unread-active-payload")
            self.assertTrue(
                blob(active).is_file(), "the unread owner's header disappeared"
            )
            del tensors
            gc.collect()
            deadline = time.monotonic() + 5
            collected = 0
            while True:
                collected += collect_in_other_process()["objects"]
                if not blob(active).exists() or time.monotonic() >= deadline:
                    break
                time.sleep(0.01)
            self.assertGreater(
                collected, 0, "last-owner release never enabled GC"
            )
            self.assertFalse(
                blob(active).exists(), "last-owner release retained the header"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
