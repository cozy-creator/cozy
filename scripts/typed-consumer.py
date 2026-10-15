#!/usr/bin/env python3
"""Read/write/recovery consumers against the installed native extension and public types."""

from __future__ import annotations

import io
from pathlib import Path
import tempfile

import tensorfs
from tensorfs import records as r
from tensorfs.derived import Config, Derivation, Part, Source, Target, Tensor

from record_contract import check


def identity(name: str) -> str:
    return tensorfs.object_id(name.encode())


def prove(root: Path) -> None:
    for name in tensorfs.__all__:
        assert hasattr(tensorfs, name), name
    store = tensorfs.Store.init(root / "source")
    raw = b"a" * 65540
    body = root / "raw"
    body.write_bytes(raw)
    check(store.put_file(body, tensorfs.object_id(raw), len(raw)), r.PutResult)
    plain = dict(tensorfs.seed_digests())["plain/1"]
    declaration = Derivation(
        sources={},
        targets={"model": Target(add={"weight": Tensor("u8", (len(raw),), plain, {"value": Part("u8", (len(raw),))})})},
        configs={"model": Config("add")},
        order=[("model", "weight")],
    )
    args = declaration.native_arguments(len(raw) + 4096)
    check(args, r.DerivationArguments)
    transaction = identity("transaction")
    work = identity("producer")
    check(store.derived_lookup(transaction), r.DerivedLookup)
    writer = store.begin_derived(transaction, 1, *args, work_fingerprint=work)
    check(store.derived_lookup(transaction), r.DerivedLookup)
    check(store.derived_write_estimate(transaction, 1, *args, work_fingerprint=work), r.DerivedWriteEstimate)
    check(writer.add_part("model", "weight", "value", io.BytesIO(raw)), r.PartReceipt)
    writer.add_config("model", io.BytesIO(b"{}"))
    check(writer.checkpoint("request", "model"), r.CheckpointReceipt)
    receipt = writer.commit()
    check(receipt, r.DerivedReceipt)
    check(store.derived_lookup(transaction), r.DerivedLookup)
    snapshot = receipt["manifest"]
    manifest = "sha256:" + snapshot["sha256"]
    row = store.manifest(manifest)
    check(row, r.ManifestRow)
    assert row["header"] is not None
    header = row["header"]
    check(tensorfs.parse_header(header), r.Header)
    check(tensorfs.fit([tensorfs.TensorRequirement("model", "weight", (len(raw),), "u8")], header, custody="canonical", encoded_leaves=False), r.FitResult)
    inspection = store.inspect_derived_source(manifest, snapshot["length"], ["model"], ["model"])
    check(inspection, r.InspectedSource)
    source = Source(manifest, snapshot["length"])
    inherited = Derivation({"base": source}, {"model": Target(source="base", source_component="model")}, {}, [("model", "weight")])
    check(inherited.native_arguments(0), r.DerivationArguments)
    plan = tensorfs.plan(header, [("model", "weight")])
    for item in plan.items():
        check(item, r.ReadItem)
    for reached in store.walk_cozytensors(manifest):
        check(reached, r.WalkObject)
    with store.acquire_cozytensors(manifest) as lease:
        received = bytearray(len(raw))
        lease.read_part_into(header, "model", "weight", "value", 0, received)
        assert received == raw
    operation = store.begin_operation("consumer", "proof", "typed")
    operation.hold_manifest(manifest, snapshot["length"])
    operation.commit_release(None, "1.0.0", "plain", manifest, snapshot["length"])
    check(store.derived_adopt(transaction, "result"), r.DispositionResult)
    check(store.derived_dispose(transaction), r.DispositionResult)
    check(tensorfs.gc(store.root), r.GCResult)
    # A result can gain fields without changing the fields this consumer understands.
    check({**receipt, "future": {"observation": 1}}, r.DerivedReceipt)
    print("typed consumer green: native content, declaration, checkpoint, inspection, read, repository and derived lifecycle")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="tensorfs-typed-consumer-") as temporary:
        prove(Path(temporary))
