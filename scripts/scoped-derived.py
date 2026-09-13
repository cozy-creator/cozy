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

import tensorfs
from tensorfs.derived import (
    Config,
    Derivation,
    DerivedTransaction,
    OutputCapability,
    Part,
    Source,
    Target,
    Tensor,
    derive,
    serve_derived,
)

WORK = "sha256:" + "ab" * 32


def identity(number):
    return "sha256:" + f"{number:064x}"


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
        native.add_part("model", "weight", "value", io.BytesIO(b"a" * size))
        source_receipt = native.commit()
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
        tx.add_part("model", "copy", "value", producer)
        assert producer.reads >= 2
        assert tx.completed_parts() == [("model", "copy", "value")]
        checkpoint = tx.checkpoint()
        assert owner.checkpoints[-1] == checkpoint
        result = tx.commit()
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
        store.derived_fence(identity(12), 1)
        try:
            owner.client.add_part("model", "weight", "value", b"x" * 4096)
        except tensorfs.errors.WriterFenced:
            pass
        else:
            raise AssertionError("stale writer accepted bytes")
        owner.finish(refused=True)

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
        writer = open_writer(store, 30, declaration(4096), 4096)
        left, right = socket.socketpair()
        child = subprocess.Popen(
            [
                sys.executable,
                __file__,
                "--isolated-client",
                store.root,
                str(left.fileno()),
            ],
            pass_fds=(left.fileno(),),
        )
        left.close()
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
        assert len(receipts) == 1
        assert store.derived_lookup(identity(30))["receipt"] == receipts[0]
        print(
            "PASS UID65533 cannot open the Store but can write its exact granted native output",
            flush=True,
        )


def main():
    if sys.argv[1:] == ["--isolation"]:
        isolation()
        return
    if len(sys.argv) == 4 and sys.argv[1] == "--isolated-client":
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
