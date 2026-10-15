#!/usr/bin/env python3
"""Real process death and empty-Store recovery through the installed wheel."""

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import tensorfs

TRANSACTION = "sha256:" + "71" * 32
FINGERPRINT = tensorfs.object_id(b"tensorfs derived resume fixture producer/1")
SHAPE = [512, 8192]
DATA = bytes([0x31]) * (SHAPE[0] * SHAPE[1])
SCALE = bytes(SHAPE[0] * 4)
MAX_NEW_BYTES = len(DATA) + len(SCALE) + 1024
ENCODING = next(digest for alias, digest in tensorfs.seed_digests() if alias == "fp8-rowwise/1")
TARGETS = {"model": {"drop": [], "add": {"weight": {
    "logical_dtype": "bf16", "shape": SHAPE, "encoding": ENCODING,
    "parts": {"data": {"dtype": "f8_e4m3fn", "shape": SHAPE},
              "scale": {"dtype": "f32", "shape": [SHAPE[0]]}},
}}}}


def begin(store, epoch, checkpoint=None):
    return store.begin_derived(TRANSACTION, epoch, {}, TARGETS,
                               {"model.json": {"kind": "add"}},
                               [("model", "weight")], MAX_NEW_BYTES,
                               work_fingerprint=FINGERPRINT, checkpoint=checkpoint)


def finish(writer):
    writer.add_part("model", "weight", "scale", io.BytesIO(SCALE))
    writer.add_config("model.json", io.BytesIO(b'{"a":1}'))
    return writer.commit()


def refused(code, call):
    try:
        call()
    except tensorfs.errors.Refusal as error:
        assert error.code == code, (code, error)
    else:
        raise AssertionError(f"expected {code}")


def child(root):
    store = tensorfs.Store.init(root)
    writer = begin(store, 7)
    writer.add_part("model", "weight", "data", io.BytesIO(DATA))
    writer.add_config("model.json", io.BytesIO(b'{"a":1}'))
    head = writer.checkpoint("request", "output")
    print(json.dumps(head), flush=True)
    class InterruptedScale:
        started = False

        def read(self, size):
            if self.started:
                # Die inside an unfinished physical role, without commit/fence/Drop.
                os._exit(23)
            self.started = True
            return SCALE[:min(size, len(SCALE) // 2)]

    writer.add_part("model", "weight", "scale", InterruptedScale())
    raise AssertionError("partial scale unexpectedly completed")


def main():
    with tempfile.TemporaryDirectory(prefix="tensorfs-derived-resume-") as area:
        area = Path(area)
        result = subprocess.run([sys.executable, __file__, "--child", str(area / "original")],
                                capture_output=True, text=True, check=False)
        assert result.returncode == 23, result.stderr
        checkpoint = json.loads(result.stdout)
        head = checkpoint["head"], checkpoint["head_length"]
        store = tensorfs.Store.open(area / "original")
        replacement = tensorfs.Store.init(area / "replacement")
        delivered = set()
        transferred = 0

        def transfer(identity, length):
            nonlocal transferred
            if identity in delivered:
                return
            body = area / "object"
            body.write_bytes(store.document(identity, length))
            replacement.put_file(body, identity, length)
            transferred += length
            delivered.add(identity)

        cursor = head
        while cursor:
            transfer(*cursor)
            offset = 0
            while True:
                page = replacement.checkpoint_page(*cursor, operation_id="request", slot="output",
                    plan_digest=checkpoint["plan_digest"], offset=offset, limit=1)
                assert len(page["objects"]) <= 1
                for obj in page["objects"]:
                    assert obj["kind"] == "blob"
                    transfer(obj["object_id"], obj["length"])
                cursor_next = page["previous"]
                offset = page["next_offset"]
                if offset is None:
                    break
            cursor = cursor_next

        writer = begin(store, 8)
        assert writer.completed_parts() == [("model", "weight", "data")]
        assert writer.completed_configs() == ["model.json"]
        detached = writer.completed_parts()
        detached.clear()
        assert writer.completed_parts() == [("model", "weight", "data")]
        assert writer.checkpoint("request", "output") == checkpoint
        refused("WRITER_FENCED", lambda: begin(store, 7))
        refused("ARTIFACT_INCOMPLETE", writer.commit)
        writer.add_part("model", "weight", "data", io.BytesIO(DATA))
        refused("TRANSACTION_CONFLICT", lambda: writer.add_part("model", "weight", "data", io.BytesIO(bytes(len(DATA)))))
        declaration = replacement.derived_declaration({}, TARGETS,
            {"model.json": {"kind": "add"}}, [("model", "weight")], MAX_NEW_BYTES,
            work_fingerprint=FINGERPRINT)
        assert replacement.derived_lookup(TRANSACTION)["state"] == "absent"
        replacement.validate_derived_checkpoint(TRANSACTION, declaration, *head,
            operation_id="request", slot="output", plan_digest=checkpoint["plan_digest"])
        assert replacement.derived_lookup(TRANSACTION)["state"] == "absent"
        refused("CROSS_SUBJECT_REPLAY", lambda: replacement.validate_derived_checkpoint(
            TRANSACTION, declaration, *head, operation_id="request", slot="other",
            plan_digest=checkpoint["plan_digest"]))
        restored = begin(replacement, 1, head)
        assert restored.completed_parts() == writer.completed_parts()
        assert restored.completed_configs() == writer.completed_configs()
        assert restored.checkpoint("request", "output") == checkpoint
        restored.add_part("model", "weight", "scale", io.BytesIO(SCALE))
        advanced = restored.checkpoint("request", "output", previous=head)
        page = replacement.checkpoint_page(advanced["head"], advanced["head_length"],
            operation_id="request", slot="output", plan_digest=checkpoint["plan_digest"])
        assert page["previous"] == head
        assert tensorfs.object_id(DATA) not in {obj["object_id"] for obj in page["objects"]}
        assert tensorfs.object_id(SCALE) in {obj["object_id"] for obj in page["objects"]}
        recovered = finish(restored)
        local = finish(writer)
        clean_store = tensorfs.Store.init(area / "clean")
        clean = begin(clean_store, 1)
        clean.add_part("model", "weight", "data", io.BytesIO(DATA))
        expected = finish(clean)
        assert recovered == local == expected
        assert recovered["declaration_digest"] == tensorfs.object_id(declaration)
        for current in [store, replacement, clean_store]:
            current.derived_dispose(TRANSACTION)
        print(f"derived resume: process death inside scale, retained {len(DATA)} B data role, "
              f"{len(delivered)} objects / {transferred} B copied to an empty Store, "
              "next checkpoint excludes prior data, higher epoch and empty Store match clean receipt")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        child(sys.argv[2])
    else:
        main()
