#!/usr/bin/env python3
"""Manual tfs-019 proof over a freshly derived, current-tree fixture.

This is deliberately not a test suite. It builds a real source manifest through `tfs`,
commits it through the repository CAS, drives the compiled Python facade, and prints
the exact committed receipt. The fixture is derived on demand so a schema change cannot
leave a stale archive looking authoritative (decision #616).
"""

from __future__ import annotations

import argparse
import io
import json
import pathlib
import re
import subprocess
from typing import Any

import tensorfs


def construction_order(
    targets: dict[str, Any], components: dict[str, dict[str, Any]]
) -> list[tuple[str, str]]:
    order: list[tuple[str, str]] = []
    for target, declaration in targets.items():
        source = declaration.get("source_component")
        dropped = set(declaration["drop"])
        if source is not None:
            order.extend(
                (target, key) for key in components[source] if key not in dropped
            )
        order.extend((target, key) for key in declaration["add"])
    return order


def run(*argv: str) -> str:
    completed = subprocess.run(argv, check=False, text=True, capture_output=True)
    print(completed.stdout, end="")
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(argv)}\n"
            f"{completed.stderr}"
        )
    return completed.stdout


def lifecycle_arms(
    store: tensorfs.Store,
    source_id: str,
    source_length: int,
    plain: str,
    components: dict[str, dict[str, Any]],
) -> None:
    sources = {"full": (source_id, source_length)}
    target: dict[str, Any] = {
        "fl2va_dit": {
            "source": "full",
            "source_component": "unet",
            "drop": [],
            "add": {
                "table": {
                    "logical_dtype": "bf16",
                    "shape": [512],
                    "encoding": plain,
                    "parts": {"value": {"dtype": "bf16", "shape": [512]}},
                }
            },
        }
    }
    copy: dict[str, Any] = {
        "fl2va_dit": {
            "source": "full",
            "source_component": "unet",
            "drop": [],
            "add": {},
        }
    }

    # A fenced id cannot reopen or write; one greater id restarts the locked declaration.
    transaction_id = "sha256:" + "a" * 64
    target_order = construction_order(target, components)
    copy_order = construction_order(copy, components)
    old = store.begin_derived(transaction_id, 1, sources, target, {}, target_order, 1024)
    if not old.fence():
        raise RuntimeError("the active writer did not fence")
    try:
        store.begin_derived(transaction_id, 1, sources, target, {}, target_order, 1024)
    except tensorfs.errors.WriterFenced:
        pass
    else:
        raise RuntimeError("a fenced WriterSessionId reopened")
    successor = store.begin_derived(
        transaction_id, 2, sources, target, {}, target_order, 1024
    )
    try:
        old.add_part("fl2va_dit", "table", "value", io.BytesIO(b"x" * 1024))
    except tensorfs.errors.WriterFenced:
        pass
    else:
        raise RuntimeError("a late fenced writer admitted bytes")
    successor.add_part(
        "fl2va_dit", "table", "value", io.BytesIO(b"x" * 1024)
    )
    receipt = successor.commit()
    commit_won = store.derived_abandon(transaction_id)
    if commit_won != {"state": "committed", "receipt": receipt}:
        raise RuntimeError("commit-versus-abandon did not return the committed receipt")
    disposition = store.derived_dispose(transaction_id)
    if store.derived_dispose(transaction_id) != disposition:
        raise RuntimeError("lost disposition reply did not replay")

    # The first declaration is a durable lock, including explicit source-component rename.
    transaction_id = "sha256:" + "b" * 64
    writer = store.begin_derived(transaction_id, 1, sources, copy, {}, copy_order, 0)
    changed: dict[str, Any] = {
        "fl2va_dit": {
            "source": "full",
            "source_component": "vae",
            "drop": [],
            "add": {},
        }
    }
    try:
        store.begin_derived(
            transaction_id,
            2,
            sources,
            changed,
            {},
            construction_order(changed, components),
            0,
        )
    except tensorfs.errors.TransactionConflict:
        pass
    else:
        raise RuntimeError("a changed declaration replay was admitted")
    writer.fence()
    if store.derived_abandon(transaction_id) != {"state": "abandoned"}:
        raise RuntimeError("fenced open transaction did not abandon")

    # Absent abandonment is a tombstone; release is separate from the immutable receipt.
    transaction_id = "sha256:" + "c" * 64
    store.derived_abandon(transaction_id)
    try:
        store.begin_derived(transaction_id, 1, sources, copy, {}, copy_order, 0)
    except tensorfs.errors.TransactionClosed:
        pass
    else:
        raise RuntimeError("an abandoned transaction reopened")

    transaction_id = "sha256:" + "d" * 64
    writer = store.begin_derived(transaction_id, 1, sources, copy, {}, copy_order, 0)
    receipt = writer.commit()
    released = store.derived_dispose(transaction_id)
    if released != {"disposition": {"kind": "released"}, "receipt": receipt}:
        raise RuntimeError("release changed the commit receipt")
    if store.derived_lookup(transaction_id) != {
        "state": "committed",
        "receipt": receipt,
    }:
        raise RuntimeError("lookup changed after release")
    print(
        "ARMS PASS fenced-reopen late-write successor changed-replay "
        "abandon-first commit-first disposition-replay release-lookup"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfs", required=True, type=pathlib.Path)
    parser.add_argument("--store", required=True, type=pathlib.Path)
    args = parser.parse_args()

    output = run(
        str(args.tfs),
        "checkpoint",
        "write",
        str(args.store),
        "--blocks",
        "22",
        "--scale",
        "32",
    )
    match = re.search(r"manifest_id\s+(sha256:[0-9a-f]{64})", output)
    if match is None:
        raise RuntimeError("checkpoint write did not print manifest_id")
    source_id = match.group(1)

    store = tensorfs.Store.open(str(args.store))
    source = store.manifest(source_id)
    source_document = source["manifest"]
    operation = store.begin_operation("derived-source", "proof", "h3-source", 1)
    operation.hold_manifest(source_id, len(source_document))
    for reached in store.walk(source_id):
        operation.hold_blob(reached["id"], reached["length"])
    evidence = json.dumps(
        {"classification_digest": tensorfs.topology_digest(source["header"])},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    operation.commit_release(
        None, "1.0.0", "local", source_id, len(source_document), evidence
    )
    header: dict[str, Any] = tensorfs.parse_header(source["header"])
    components: dict[str, dict[str, Any]] = header["components"]
    source_dit = components["unet"]
    drops = sorted(source_dit)[:106]
    if len(drops) != 106:
        raise RuntimeError(f"source fixture exposed {len(drops)} drop keys, expected 106")
    plain = next(
        tensor["encoding"]
        for tensor in source_dit.values()
        if list(tensor["parts"]) == ["value"]
    )
    additions: dict[str, Any] = {
        f"adaln.table.{index:03d}": {
            "logical_dtype": "bf16",
            "shape": [512],
            "encoding": plain,
            "parts": {"value": {"dtype": "bf16", "shape": [512]}},
        }
        for index in range(51)
    }
    targets: dict[str, Any] = {
        "fl2va_dit": {
            "source": "full",
            "source_component": "unet",
            "drop": drops,
            "add": additions,
        },
        "ref2va_dit": {
            "source": "full",
            "source_component": "unet",
            "drop": [],
            "add": {},
        },
        "text_encoder": {
            "source": "full",
            "source_component": "text_encoder",
            "drop": [],
            "add": {},
        },
        "video_vae": {
            "source": "full",
            "source_component": "vae",
            "drop": [],
            "add": {},
        },
        "audio_vae": {
            "source": "full",
            "source_component": "vae",
            "drop": [],
            "add": {},
        },
    }
    configs: dict[str, Any] = {
        "dit": {"source": "full", "source_config": "unet"},
        "text_encoder": {
            "source": "full",
            "source_config": "text_encoder",
        },
    }
    transaction_id = tensorfs.object_id(f"tfs-019:{source_id}".encode())
    writer = store.begin_derived(
        transaction_id,
        1,
        {"full": (source_id, len(source_document))},
        targets,
        configs,
        construction_order(targets, components),
        51 * 1024,
    )
    consumed_selector: tuple[str, str, int] | None = None
    for dropped_key in drops:
        for role, part in source_dit[dropped_key]["parts"].items():
            segments = part.get("segments", [])
            if segments:
                consumed_selector = (dropped_key, role, segments[0]["length"])
                break
        if consumed_selector is not None:
            break
    if consumed_selector is None:
        raise RuntimeError("the derived source fixture has no dropped object-backed role")
    consumed_key, consumed_role, consumed_length = consumed_selector
    consumed = bytearray(consumed_length)
    writer.source_read_into(
        "full",
        "unet",
        consumed_key,
        consumed_role,
        0,
        consumed,
    )
    inline_selector: tuple[str, str, int] | None = None
    for dropped_key in drops:
        for role, part in source_dit[dropped_key]["parts"].items():
            if "inline" in part:
                inline_length = tensorfs.DTYPES[part["dtype"]]
                for dimension in part["shape"]:
                    inline_length *= dimension
                inline_selector = (dropped_key, role, inline_length)
                break
        if inline_selector is not None:
            break
    if inline_selector is None:
        raise RuntimeError("the derived source fixture has no dropped inline role")
    inline_key, inline_role, inline_length = inline_selector
    inline = bytearray(inline_length)
    writer.source_read_into("full", "unet", inline_key, inline_role, 0, inline)
    for index, tensor_key in enumerate(sorted(additions)):
        writer.add_part(
            "fl2va_dit",
            tensor_key,
            "value",
            io.BytesIO(bytes([index]) * 1024),
        )
    receipt = writer.commit()
    before = store.derived_lookup(transaction_id)
    disposed = store.derived_dispose(transaction_id)
    after = store.derived_lookup(transaction_id)

    if before["receipt"] != receipt or after["receipt"] != receipt:
        raise RuntimeError("commit receipt changed across lookup/disposition")
    if disposed["receipt"] != receipt:
        raise RuntimeError("disposition returned a different commit receipt")
    observation = receipt["inherit_observation"]
    if observation["reads"] != 0 or observation["hashes"] != 0:
        raise RuntimeError(f"inherited payload bytes moved: {observation}")
    target = store.manifest(f"sha256:{receipt['manifest']['sha256']}")
    evidence = json.loads(receipt["checkpoint_evidence_canonical_bytes"])
    if evidence != {
        "classification_digest": tensorfs.topology_digest(target["header"])
    }:
        raise RuntimeError(
            "derived CheckpointEvidence is not the exact minimal final-topology fact"
        )
    target_header: dict[str, Any] = tensorfs.parse_header(target["header"])
    target_components: dict[str, dict[str, Any]] = target_header["components"]
    if sorted(target_components) != [
        "audio_vae",
        "fl2va_dit",
        "ref2va_dit",
        "text_encoder",
        "video_vae",
    ]:
        raise RuntimeError(f"wrong target components: {sorted(target_components)}")
    fl2va = target_components["fl2va_dit"]
    if any(key in fl2va for key in drops):
        raise RuntimeError("a dropped tensor survived with one of its roles")
    if sum(key.startswith("adaln.table.") for key in fl2va) != 51:
        raise RuntimeError("the 51 declared table tensors were not all added")

    lifecycle_arms(store, source_id, len(source_document), plain, components)

    print(
        "DERIVED PASS "
        f"source_tensors={sum(len(v) for v in components.values())} "
        "drop=106 add=51 components=5 "
        f"inherited_objects={observation['objects']} "
        f"inherited_bytes={observation['bytes']} "
        f"consumed_source_bytes={len(consumed) + len(inline)} "
        "source_reader=segmented+inline "
        "inherited_reads=0 inherited_hashes=0 immutable_receipt=1 "
        "checkpoint_evidence=minimal"
    )


if __name__ == "__main__":
    main()
