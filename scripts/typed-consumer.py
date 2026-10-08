#!/usr/bin/env python3
"""Read/write/recovery consumers against the installed native extension and public types."""

from __future__ import annotations

import io
import json
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
    assert tensorfs.TensorPart is r.TensorPart
    store = tensorfs.Store.init(root / "source")
    raw = b"a" * 65540
    original = store.put_reader(io.BytesIO(raw), tensorfs.object_id(raw), len(raw))
    check(original, r.PutResult)
    check(store.verify(original["id"]), r.VerifyResult)
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
    check(store.derived_abandon(transaction), r.AbandonResult)
    snapshot = receipt["manifest"]
    manifest = "sha256:" + snapshot["sha256"]
    row = store.manifest(manifest)
    check(row, r.ManifestRow)
    check(tensorfs.parse_manifest(row["manifest"]), r.ManifestDocument)
    assert row["header"] is not None
    header = row["header"]
    check(tensorfs.parse_header(header), r.Header)
    check(tensorfs.run_doc("manifest", row["manifest"]), r.DocumentResult)
    check(tensorfs.capability("plain/1", "cpu"), r.CapabilityResult)
    check(tensorfs.fit([tensorfs.TensorRequirement("model", "weight", (len(raw),), "u8")], header, custody="canonical", encoded_leaves=False), r.FitResult)
    inspection = store.inspect_derived_source(manifest, snapshot["length"], ["model"], ["model"])
    check(inspection, r.InspectedSource)
    source = Source(manifest, snapshot["length"])
    inherited = Derivation({"base": source}, {"model": Target(source="base", source_component="model")}, {}, [("model", "weight")])
    inherited_args = inherited.native_arguments(0)
    check(inherited_args, r.DerivationArguments)
    child = store.begin_derived(identity("child"), 1, *inherited_args, work_fingerprint=work)
    for facts in child.sources():
        check(facts, r.SourceFacts)
    child.fence()
    check(store.derived_abandon(identity("child")), r.AbandonResult)
    plan = tensorfs.plan(header, [("model", "weight")])
    for item in plan.items():
        check(item, r.ReadItem)
    with store.acquire_cozytensors(manifest) as lease:
        for observation in lease.receipts:
            check(observation, r.ReadReceipt)
        slots = [bytearray(1 << 17)]
        received = bytearray()

        def consume(batch: tensorfs.Batch) -> None:
            for item in batch.items():
                check(item, r.BatchItem)
                received.extend(slots[batch.slot][item["off"]:item["off"] + item["len"]])
            batch.release()

        check(lease.stream(plan, slots, consume), r.StreamResult)
        assert received == raw
    closure = store.walk_cozytensors(manifest)
    for reached in closure:
        check(reached, r.WalkObject)
    destination = tensorfs.Store.init(root / "destination")
    operation = destination.begin_operation("consumer", "proof", "typed")
    operation.hold_manifest(manifest, snapshot["length"])
    for reached in closure:
        operation.hold_blob(reached["id"], reached["length"])
        check(tensorfs.transfer_object(store, destination, reached["id"], reached["length"]), r.PutResult)
    check(tensorfs.transfer_manifest(store, destination, manifest, snapshot["length"]), r.PutResult)
    fetch = tensorfs.fetch_plan(destination, "consumer", manifest, snapshot["length"], [(item["id"], item["length"]) for item in closure])
    check(fetch, r.FetchPlan)
    repository = operation.commit_release(None, "1.0.0", "plain", manifest, snapshot["length"])
    check(destination.resolve_release("proof", "typed", "1.0.0", "plain"), r.ReleaseRow)
    assert repository == destination.repo_get("proof", "typed")
    local = store.replace_local(None, "typed", identity("selection"), manifest, snapshot["length"])
    check(store.resolve_local("typed"), r.LocalRow)
    check(store.remove_local(local, "typed"), r.RemovedRepository)
    check(store.derived_adopt(transaction, "result"), r.DispositionResult)
    receipt_digest = tensorfs.object_id(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode())
    retained = store.retain_derived_result(transaction, receipt_digest, identity("recipient"))
    check(retained, r.RetainedResult)
    check(store.derived_dispose(transaction), r.DispositionResult)
    check(store.release_derived_retention(transaction, receipt_digest, identity("recipient")), r.RetainedResult)
    check(tensorfs.gc(store.root), r.GCResult)
    check(tensorfs.relieve(destination), r.Relief)
    check(tensorfs.stats(), r.Stats)
    # A result can gain fields without changing the fields this consumer understands.
    check({**receipt, "future": {"observation": 1}}, r.DerivedReceipt)
    print("typed consumer green: native content, declaration, checkpoint, inspection, read, repository, transfer and retained-result lifecycle")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="tensorfs-typed-consumer-") as temporary:
        prove(Path(temporary))
