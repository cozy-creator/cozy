#!/usr/bin/env python3
"""Actual native writers behind scoped FD handles, including lazy nested reads."""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
from typing import get_type_hints

import tensorfs
from tensorfs.derived import (
    Config,
    Derivation,
    DerivedTransaction,
    OutputCapability,
    Part,
    Source,
    SourceCapability,
    SourceInspection,
    Target,
    Tensor,
    derive,
    serve_derived,
    serve_derived_replay,
    serve_source,
)
from tensorfs._derived_channel import _Channel
from tensorfs.records import (
    CheckpointReceipt, DerivedReceipt, DerivedSource, InheritObservation,
    ObjectRef, PartReceipt,
)

WORK = "sha256:" + "ab" * 32


def identity(number):
    return "sha256:" + f"{number:064x}"


def record_matches(record, schema):
    """Check public annotations against native-produced facts; additive fields are allowed."""
    assert schema.__required_keys__ <= record.keys(), (schema.__name__, record)
    for name, kind in get_type_hints(schema).items():
        if kind in (str, int):
            assert type(record[name]) is kind, (name, record[name])
    return record


def receipt_matches(receipt):
    record_matches(receipt, DerivedReceipt)
    record_matches(receipt["inherit_observation"], InheritObservation)
    for obj in [receipt["header"], receipt["manifest"], *receipt["added_objects"]]:
        record_matches(obj, ObjectRef)
    for source in receipt["sources"]:
        record_matches(source, DerivedSource)
        record_matches(source["header"], ObjectRef)
        record_matches(source["manifest"], ObjectRef)


def declaration(length, source=None):
    plain = next(value for name, value in tensorfs.seed_digests() if name == "plain/1")
    return Derivation(
        sources={"base": source} if source else {},
        targets={
            "model": Target(
                source="base" if source else "",
                source_component="model" if source else "",
                add={
                    "copy" if source else "weight": Tensor(
                        "u8",
                        (length,),
                        plain,
                        {"value": Part("u8", (length,))},
                    )
                },
            )
        },
        configs={},
        order=[("model", "weight"), ("model", "copy")]
        if source
        else [("model", "weight")],
    )


class Owner:
    def __init__(self, writer, *, record=None):
        client, server = socket.socketpair()
        self.client = DerivedTransaction(client.detach())
        self.stop_socket = server.dup()
        self.errors = []
        self.receipts = []
        self.checkpoints = []
        self.admissions = []
        exclusion = threading.Lock()

        @contextlib.contextmanager
        def admit(operation, bound):
            assert exclusion.acquire(blocking=False), (
                "nested storage admission deadlocked"
            )
            self.admissions.append((operation, bound))
            try:
                yield
            finally:
                exclusion.release()

        def accepted(receipt):
            if record:
                record(receipt)
            self.receipts.append(receipt)

        def run():
            try:
                serve_derived(
                    writer,
                    server.detach(),
                    operation_id="proof",
                    slot="model",
                    checkpoint=None,
                    check_current=lambda: None,
                    admit_io=admit,
                    record_checkpoint=self.checkpoints.append,
                    record_receipt=accepted,
                )
            except Exception as error:
                self.errors.append(error)

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def finish(self, *, refused=False):
        self.client.close()
        self.thread.join(10)
        self.stop_socket.close()
        assert not self.thread.is_alive(), "native writer channel did not stop"
        assert bool(self.errors) == refused, self.errors


def open_writer(store, number, plan, maximum, epoch=1):
    return store.begin_derived(
        identity(number), epoch, *plan.native_arguments(maximum), work_fingerprint=WORK
    )


def exercise():
    with tempfile.TemporaryDirectory(prefix="tensorfs-scoped-derived-") as root:
        store = tensorfs.Store.init(Path(root) / "store")
        size = 2 << 20
        original = declaration(size)
        native = open_writer(store, 1, original, size)
        part = native.add_part("model", "weight", "value", io.BytesIO(b"a" * size))
        record_matches(part, PartReceipt)
        source_receipt = native.commit()
        receipt_matches(source_receipt)
        assert tensorfs.derived.ObjectRef is ObjectRef
        assert tensorfs.derived.DerivedReceipt is DerivedReceipt
        ref = source_receipt["manifest"]
        plan = declaration(size, Source("sha256:" + ref["sha256"], ref["length"]))
        assert (
            json.loads(
                store.derived_declaration(
                    *plan.native_arguments(size), work_fingerprint=WORK
                )
            )["work_fingerprint"]
            == WORK
        )
        owner = Owner(open_writer(store, 2, plan, size))
        output = OutputCapability(
            lambda incoming: owner.client if incoming is plan else None
        )
        tx = derive(output, plan)

        class Lazy:
            offset = 0
            reads = 0

            def read(self, maximum):
                length = min(maximum, size - self.offset)
                if not length:
                    return b""
                into = bytearray(length)
                tx.source_read_into(
                    "base", "model", "weight", "value", self.offset, into
                )
                assert into == b"a" * length
                self.offset += length
                self.reads += 1
                return bytes(value ^ 3 for value in into)

        producer = Lazy()
        part = tx.add_part("model", "copy", "value", producer)
        record_matches(part, PartReceipt)
        assert producer.reads >= 2
        assert tx.completed_parts() == [("model", "copy", "value")]
        checkpoint = tx.checkpoint()
        record_matches(checkpoint, CheckpointReceipt)
        assert checkpoint["head"] is None or isinstance(checkpoint["head"], str)
        assert owner.checkpoints[-1] == checkpoint
        result = tx.commit()
        receipt_matches(result)
        assert tx.commit() is result
        newer = {**result, "future_observation": {"kept": True}}
        assert DerivedTransaction.replay(newer).commit() == newer
        assert owner.receipts == [result], (
            "commit returned before owner recorded receipt"
        )
        assert [op for op, _ in owner.admissions].count("read") == 0, (
            "nested read reacquired admission"
        )
        owner.finish()
        reference = open_writer(store, 3, plan, size)
        reference.add_part("model", "copy", "value", io.BytesIO(b"b" * size))
        assert reference.commit()["manifest"] == result["manifest"]
        assert result["inherit_observation"] == {
            "bytes": size,
            "objects": 1,
            "reads": 0,
            "hashes": 0,
        }
        store.derived_dispose(identity(1))
        tensorfs.gc(store.root)
        store.inspect_derived_source(
            "sha256:" + result["manifest"]["sha256"],
            result["manifest"]["length"],
            ["model"],
            [],
        )
        print(
            "PASS typed declaration + native exact-result parity + nested lazy source reads + receipt ACK + zero-copy inheritance/GC",
            flush=True,
        )

        small = declaration(4096)
        for number, payload in ((10, b"x" * 100), (11, b"x" * 4097)):
            owner = Owner(open_writer(store, number, small, 4096))
            try:
                owner.client.add_part("model", "weight", "value", payload)
            except tensorfs.errors.Refusal:
                pass
            else:
                raise AssertionError("wrong-length producer was accepted")
            owner.finish(refused=True)
            reopened = open_writer(store, number, small, 4096, epoch=2)
            assert reopened.completed_parts() == []
            reopened.fence()

        owner = Owner(open_writer(store, 12, small, 4096))
        fencer = open_writer(store, 12, small, 4096, epoch=2)  # a higher epoch fences the first
        try:
            owner.client.add_part("model", "weight", "value", b"x" * 4096)
        except tensorfs.errors.WriterFenced:
            pass
        else:
            raise AssertionError("stale writer accepted bytes")
        owner.finish(refused=True)
        fencer.fence()

        def fail_record(receipt):
            raise RuntimeError("injected owner ACK failure")

        owner = Owner(open_writer(store, 13, small, 4096), record=fail_record)
        owner.client.add_part("model", "weight", "value", memoryview(b"x" * 4096))
        try:
            owner.client.commit()
        except Exception:
            pass
        else:
            raise AssertionError("commit escaped failed owner ACK")
        owner.finish(refused=True)
        committed = store.derived_lookup(identity(13))
        assert (
            committed["state"] == "committed"
            and committed["disposition"]["kind"] == "pending"
        )
        assert (
            DerivedTransaction.replay(committed["receipt"]).commit()
            == committed["receipt"]
        )
        print(
            "PASS short/long stream, stale writer, native commit-before-ACK custody and replay",
            flush=True,
        )

        # Malformed/overbroad commands cannot select another transaction or Store.
        from tensorfs._derived_channel import _Channel

        for number, frame in (
            (14, {"op": "dispose", "transaction": identity(13)}),
            (
                15,
                {
                    "op": "part",
                    "component": "model",
                    "key": "weight",
                    "role": "value",
                    "epoch": 9,
                },
            ),
        ):
            owner = Owner(open_writer(store, number, small, 4096))
            wire = owner.client._channel
            assert isinstance(wire, _Channel)
            wire.send(frame)
            try:
                wire.reply()
            except Exception:
                pass
            else:
                raise AssertionError("foreign authority command was accepted")
            owner.finish(refused=True)
        assert store.derived_lookup(identity(13))["disposition"]["kind"] == "pending"

        # A producer exception disconnects mid-native-read; the incomplete part is not accepted.
        owner = Owner(open_writer(store, 16, original, size))

        class Interrupted:
            calls = 0

            def read(self, maximum):
                self.calls += 1
                if self.calls == 1:
                    return b"x" * min(maximum, 65536)
                raise RuntimeError("injected producer interruption")

        try:
            owner.client.add_part("model", "weight", "value", Interrupted())
        except RuntimeError:
            pass
        else:
            raise AssertionError("interrupted producer returned success")
        owner.finish()
        replacement = open_writer(store, 16, original, size, epoch=2)
        assert replacement.completed_parts() == []
        replacement.fence()
        print(
            "PASS overbroad command/foreign epoch refusal and interrupted partial-stream recovery",
            flush=True,
        )

        # Authority revocation must wake an owner blocked on a stalled producer,
        # rather than wait for another frame before checking its grant again.
        owner = Owner(open_writer(store, 19, small, 4096))
        waiting, resume = threading.Event(), threading.Event()
        client_errors = []

        class Stalled:
            def read(self, maximum):
                waiting.set()
                assert resume.wait(10), "cancellation test failed to unblock producer"
                return b"x" * min(maximum, 4096)

        def stalled_client():
            try:
                owner.client.add_part("model", "weight", "value", Stalled())
            except Exception as error:
                client_errors.append(error)

        producer_thread = threading.Thread(target=stalled_client, daemon=True)
        producer_thread.start()
        assert waiting.wait(10), "native owner did not request producer bytes"
        owner.stop_socket.shutdown(socket.SHUT_RDWR)
        owner.thread.join(10)
        assert not owner.thread.is_alive(), "revoked owner stayed blocked in recv"
        resume.set()
        producer_thread.join(10)
        assert not producer_thread.is_alive() and client_errors
        owner.finish()
        replacement = open_writer(store, 19, small, 4096, epoch=2)
        assert replacement.completed_parts() == []
        replacement.fence()
        print("PASS owner shutdown fences a stalled producer without waiting for bytes", flush=True)

        # Partial role custody is resumable; only a complete data+scale pair commits.
        fp8 = next(
            value for name, value in tensorfs.seed_digests() if name == "fp8-rowwise/1"
        )
        quantized = Derivation(
            {},
            {
                "model": Target(
                    add={
                        "weight": Tensor(
                            "bf16",
                            (32, 128),
                            fp8,
                            {
                                "data": Part("f8_e4m3fn", (32, 128)),
                                "scale": Part("f32", (32,)),
                            },
                        )
                    }
                )
            },
            {},
            [("model", "weight")],
        )
        owner = Owner(open_writer(store, 17, quantized, 4224))
        owner.client.add_part("model", "weight", "data", bytes(4096))
        assert owner.client.checkpoint()["head"] is not None
        try:
            owner.client.commit()
        except tensorfs.errors.Refusal:
            pass
        else:
            raise AssertionError(
                "incomplete FP8 data+scale pair became a completed artifact"
            )
        owner.finish(refused=True)
        assert store.derived_lookup(identity(17))["state"] == "open"
        owner = Owner(open_writer(store, 17, quantized, 4224, epoch=2))
        assert owner.client.completed_parts() == [("model", "weight", "data")]
        owner.client.add_part("model", "weight", "scale", bytes(128))
        assert owner.client.checkpoint()["head"] is not None
        owner.client.commit()
        owner.finish()
        print(
            "PASS partial data-role recovery and atomic complete data-plus-scale artifact commit",
            flush=True,
        )

        # The owner admits the native bound before pulling config bytes. Whitespace
        # that would canonicalize to a tiny object must not escape that read budget.
        config_plan = declaration(4096)
        config_plan = Derivation(
            config_plan.sources,
            config_plan.targets,
            {"model.json": Config("add")},
            config_plan.order,
        )
        native = open_writer(store, 18, config_plan, 4096)
        assert native.config_write_bound("model.json") == 4096
        owner = Owner(native)
        oversized = io.BytesIO(b" " * 8192 + b"{}")
        try:
            owner.client.add_config("model.json", oversized)
        except tensorfs.errors.SizeCap:
            pass
        else:
            raise AssertionError("config exceeded its native admitted read bound")
        assert oversized.tell() == 4097, oversized.tell()
        owner.finish(refused=True)
        owner = Owner(open_writer(store, 18, config_plan, 4096, epoch=2))
        assert owner.client.completed_configs() == []
        owner.client.add_config("model.json", b'{"a":1}')
        assert owner.client.completed_configs() == ["model.json"]
        owner.finish()
        print("PASS config pull obeys the native admitted byte bound", flush=True)

        # Inspection uses the same native parser before a target plan exists.
        source_writer = open_writer(store, 21, config_plan, 4103)
        source_writer.add_part("model", "weight", "value", io.BytesIO(bytes(4096)))
        source_writer.add_config("model.json", io.BytesIO(b'{"a":1}'))
        source_ref = source_writer.commit()["manifest"]
        selected = Source("sha256:" + source_ref["sha256"], source_ref["length"])

        def inspect_bound_source(components, configs):
            native = store.inspect_derived_source(
                selected.manifest, selected.length, components, configs
            )
            projected = SourceInspection.from_native(native)
            if configs:
                assert projected.configs["model.json"] is native["configs"][0]["data"]
            return projected

        capability = SourceCapability(
            selected.manifest, selected.length, inspect_bound_source
        )
        inspected = capability.inspect(components=["model"], configs=["model.json"])
        assert inspected.source == selected
        assert inspected.configs == {"model.json": b'{"a":1}'}
        assert inspected.components["model"]["weight"] == config_plan.targets["model"].add["weight"]
        assert declaration(4096, capability).native_arguments(4096) == declaration(
            4096, selected
        ).native_arguments(4096)
        try:
            capability.inspect(components=["absent"])
        except tensorfs.errors.Refusal:
            pass
        else:
            raise AssertionError("native inspection accepted an absent component")
        print("PASS typed source inspection preserves native identity, geometry and config bytes", flush=True)

        # Source metadata/config bytes use the exact-source descriptor as well;
        # the client need not open a Store, even for a config larger than control.
        large_config = b'{"text":"' + b"x" * (2 << 20) + b'"}'
        writer = open_writer(store, 22, config_plan, 4096 + len(large_config))
        writer.add_part("model", "weight", "value", io.BytesIO(bytes(4096)))
        writer.add_config("model.json", io.BytesIO(large_config))
        large_ref = writer.commit()["manifest"]
        fixed = Source("sha256:" + large_ref["sha256"], large_ref["length"])
        left, right = socket.socketpair()
        failures = []

        def inspect_owner():
            try:
                serve_source(store, right.detach(), fixed.manifest, fixed.length)
            except Exception as error:
                failures.append(error)

        inspector = threading.Thread(target=inspect_owner, daemon=True)
        inspector.start()
        with SourceCapability.from_fd(fixed.manifest, fixed.length, left.detach()) as source:
            actual = source.inspect()
            assert actual.source == fixed
            assert actual.configs == {"model.json": large_config}
            assert actual.components["model"]["weight"].shape == (4096,)
        inspector.join(10)
        assert not inspector.is_alive() and not failures

        replay_receipt = store.derived_lookup(identity(22))["receipt"]
        recorded = []
        left, right = socket.socketpair()

        def replay_owner():
            serve_derived_replay(replay_receipt, right.detach(), check_current=lambda: None,
                                 record_receipt=recorded.append)

        replay_thread = threading.Thread(target=replay_owner, daemon=True)
        replay_thread.start()
        replayed = DerivedTransaction.accept(left.detach())
        assert replayed.receipt == replay_receipt and replayed.commit() == replay_receipt
        assert recorded == [replay_receipt]
        replay_thread.join(10)
        assert not replay_thread.is_alive()
        owner = Owner(open_writer(store, 23, small, 4096))
        assert owner.client._call("receipt") is None
        owner.finish()
        print("PASS exact-source descriptor, multi-frame config and recorded native replay handoff", flush=True)

        # A newer owner may add fields to its answers; this client reads the ones it uses.
        left, right = socket.socketpair()

        def newer_owner():
            channel = _Channel(right.detach())
            channel.receive()
            channel.send({"result": None, "served_by": "newer-owner"})
            channel.receive()
            channel.send({"error": "TRANSACTION_CONFLICT", "detail": "fenced", "retry_after": 3})
            channel.close()

        newer = threading.Thread(target=newer_owner, daemon=True)
        newer.start()
        client = DerivedTransaction(left.detach())
        assert client._call("receipt") is None
        try:
            client._call("checkpoint")
        except tensorfs.errors.Refusal as exc:
            assert exc.code == "TRANSACTION_CONFLICT"
        else:
            raise AssertionError("an owner's refusal with an extra field was not raised")
        newer.join(10)
        assert not newer.is_alive()
        print("PASS additive answer fields from a newer owner are ignored", flush=True)

        # Selection is a set; inspection preserves the artifact's constructor order.
        plain = dict(tensorfs.seed_digests())["plain/1"]
        ordered_plan = Derivation({}, {
            "z-last": Target(add={"z-weight": Tensor("u8", (8,), plain, {"value": Part("u8", (8,))})}),
            "a-first": Target(add={"a-weight": Tensor("u8", (8,), plain, {"value": Part("u8", (8,))})}),
        }, {}, [("z-last", "z-weight"), ("a-first", "a-weight")])
        writer = open_writer(store, 24, ordered_plan, 16)
        writer.add_part("z-last", "z-weight", "value", io.BytesIO(bytes(8)))
        writer.add_part("a-first", "a-weight", "value", io.BytesIO(bytes(8)))
        ordered = writer.commit()["manifest"]
        ordered_source = Source("sha256:" + ordered["sha256"], ordered["length"])
        native_order = store.inspect_derived_source(ordered_source.manifest, ordered_source.length, ["a-first", "z-last"], [])
        assert [row["name"] for row in native_order["components"]] == ["z-last", "a-first"]
        left, right = socket.socketpair()
        refused_components = []
        def serve_ordered_source():
            try:
                serve_source(store, right.detach(), ordered_source.manifest, ordered_source.length)
            except tensorfs.errors.Refusal as exc:
                refused_components.append(exc.code)
        ordered_thread = threading.Thread(target=serve_ordered_source, daemon=True)
        ordered_thread.start()
        with SourceCapability.from_fd(ordered_source.manifest, ordered_source.length, left.detach()) as ordered_capability:
            ordered_view = ordered_capability.inspect(components=["a-first", "z-last"])
            try:
                ordered_capability.inspect(components=["z-last", "absent"])
            except tensorfs.errors.Refusal as exc:
                assert exc.code == "MISSING_FIELD"
            else:
                raise AssertionError("scoped inspection silently omitted an absent component")
        ordered_thread.join(10)
        assert not ordered_thread.is_alive()
        assert refused_components == ["MISSING_FIELD"]
        assert tuple((component, key) for component, rows in ordered_view.components.items() for key in rows) == tuple(ordered_plan.order)
        print("PASS native source inspection and FD selection preserve nonlexical construction order", flush=True)

        # Real process exit at native commit; no in-memory callback state survives.
        left, right = socket.socketpair()
        child = subprocess.Popen(
            [
                sys.executable,
                __file__,
                "--crash-owner",
                store.root,
                str(right.fileno()),
            ],
            pass_fds=(right.fileno(),),
        )
        right.close()
        client = DerivedTransaction(left.detach())
        client.add_part("model", "weight", "value", b"z" * 4096)
        try:
            client.commit()
        except (EOFError, OSError):
            pass
        else:
            raise AssertionError("dead owner acknowledged commit")
        assert child.wait(10) == 91
        recovered = tensorfs.Store.open(store.root).derived_lookup(identity(20))
        assert (
            recovered["state"] == "committed"
            and recovered["disposition"]["kind"] == "pending"
        )
        tensorfs.gc(store.root)
        receipt = recovered["receipt"]
        store.inspect_derived_source(
            "sha256:" + receipt["manifest"]["sha256"],
            receipt["manifest"]["length"],
            ["model"],
            [],
        )
        print(
            "PASS real owner death after native commit; pending output survives reopen and actual GC",
            flush=True,
        )


def isolation():
    assert os.getuid() == 0, (
        "the isolated UID proof requires a root-owned fixture process"
    )
    with tempfile.TemporaryDirectory(prefix="tensorfs-scoped-uid-") as root:
        store = tensorfs.Store.init(Path(root) / "store")
        config_bytes = b'{"text":"' + b"x" * (2 << 20) + b'"}'
        plan = declaration(4096)
        configured = Derivation(plan.sources, plan.targets, {"model.json": Config("add")}, plan.order)
        source_writer = open_writer(store, 31, configured, 4096 + len(config_bytes))
        source_writer.add_part("model", "weight", "value", io.BytesIO(bytes(4096)))
        source_writer.add_config("model.json", io.BytesIO(config_bytes))
        source_receipt = source_writer.commit()
        manifest = source_receipt["manifest"]
        source = Source("sha256:" + manifest["sha256"], manifest["length"])
        writer = open_writer(store, 30, declaration(4096), 4096)
        left, right = socket.socketpair()
        source_left, source_right = socket.socketpair()
        replay_left, replay_right = socket.socketpair()
        replay_recorded = []
        errors = []

        def inspect_owner():
            try:
                serve_source(store, source_right.detach(), source.manifest, source.length)
            except Exception as exc:
                errors.append(exc)

        def replay_owner():
            try:
                serve_derived_replay(source_receipt, replay_right.detach(),
                                     check_current=lambda: None, record_receipt=replay_recorded.append)
            except Exception as exc:
                errors.append(exc)

        child = subprocess.Popen(
            [
                sys.executable,
                __file__,
                "--isolated-client",
                store.root,
                str(left.fileno()),
                str(source_left.fileno()),
                source.manifest,
                str(source.length),
                str(replay_left.fileno()),
            ],
            pass_fds=(left.fileno(), source_left.fileno(), replay_left.fileno()),
        )
        left.close()
        source_left.close()
        replay_left.close()
        inspector = threading.Thread(target=inspect_owner, daemon=True)
        replayer = threading.Thread(target=replay_owner, daemon=True)
        inspector.start()
        replayer.start()
        receipts = []
        serve_derived(
            writer,
            right.detach(),
            operation_id="proof",
            slot="model",
            checkpoint=None,
            check_current=lambda: None,
            admit_io=lambda *_: contextlib.nullcontext(),
            record_checkpoint=lambda _: None,
            record_receipt=receipts.append,
        )
        assert child.wait(10) == 0
        inspector.join(10)
        replayer.join(10)
        assert not inspector.is_alive() and not replayer.is_alive() and not errors
        assert len(receipts) == 1
        assert store.derived_lookup(identity(30))["receipt"] == receipts[0]
        assert replay_recorded == [source_receipt]
        print(
            "PASS UID65533 cannot open the Store but can inspect its exact source, read a 2MiB config, write its granted output and receive recorded native replay",
            flush=True,
        )


def main():
    if sys.argv[1:] == ["--isolation"]:
        isolation()
        return
    if len(sys.argv) == 8 and sys.argv[1] == "--isolated-client":
        # CI's interpreter can live below a private runner directory. Bootstrap
        # imports first, then irreversibly drop every UID/group before opening any
        # Store or using the exact inherited writer FD. No Store handle crosses.
        os.setgroups([])
        os.setgid(65533)
        os.setuid(65533)
        assert os.getresuid() == (65533, 65533, 65533)
        assert os.getresgid() == (65533, 65533, 65533)
        assert os.getgroups() == []
        try:
            tensorfs.Store.open(sys.argv[2])
        except tensorfs.errors.Refusal:
            pass
        else:
            raise AssertionError("isolated client opened the privileged Store")
        with SourceCapability.from_fd(sys.argv[5], int(sys.argv[6]), int(sys.argv[4])) as source:
            actual = source.inspect()
            assert actual.configs["model.json"] == b'{"text":"' + b"x" * (2 << 20) + b'"}'
            assert actual.components["model"]["weight"].shape == (4096,)
        replayed = DerivedTransaction.accept(int(sys.argv[7]))
        assert replayed.receipt is not None
        assert "sha256:" + replayed.receipt["manifest"]["sha256"] == sys.argv[5]
        with DerivedTransaction(int(sys.argv[3])) as client:
            client.add_part("model", "weight", "value", b"q" * 4096)
            client.checkpoint()
            client.commit()
        return
    if len(sys.argv) == 4 and sys.argv[1] == "--crash-owner":
        store = tensorfs.Store.open(sys.argv[2])
        writer = open_writer(store, 20, declaration(4096), 4096)
        serve_derived(
            writer,
            int(sys.argv[3]),
            operation_id="proof",
            slot="model",
            checkpoint=None,
            check_current=lambda: None,
            admit_io=lambda *_: contextlib.nullcontext(),
            record_checkpoint=lambda _: None,
            record_receipt=lambda _: os._exit(91),
        )
        raise AssertionError("crash owner returned")
    exercise()


if __name__ == "__main__":
    main()
