#!/usr/bin/env python3
"""Exact source-part reads and encoded equality through real isolated descriptors."""
from __future__ import annotations

import contextlib
import hashlib
import io
import os
import socket
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import tensorfs
from tensorfs.derived import Derivation, Part, Source, SourceCapability, Target, Tensor, serve_source
from tensorfs._derived_channel import ChannelRefusal


def fixture(store, number=1, changed=False):
    data = bytes(range(256)) * 8192
    if changed:
        data = b"x" + data[1:]
    plain = dict(tensorfs.seed_digests())["plain/1"]
    tensor = Tensor("f32", (512, 1024), plain, {"value": Part("f32", (512, 1024))})
    definition = Derivation({}, {"zeta": Target(add={"weight": tensor}), "alpha": Target(add={"renamed": tensor})}, {}, [("zeta", "weight"), ("alpha", "renamed")])
    transaction = tensorfs.object_id(f"reader-source-{number}".encode())
    writer = store.begin_derived(transaction, 1, *definition.native_arguments(len(data) * 2), work_fingerprint=tensorfs.object_id(b"reader-fixture"))
    writer.add_part("zeta", "weight", "value", io.BytesIO(data))
    writer.add_part("alpha", "renamed", "value", io.BytesIO(data))
    receipt = writer.commit()
    manifest = receipt["manifest"]
    return Source("sha256:" + manifest["sha256"], manifest["length"]), transaction, data


@contextlib.contextmanager
def source_handle(store, source, *, claimed=None, current=lambda: None):
    client, owner = socket.socketpair()
    errors = []
    def serve():
        try:
            serve_source(store, owner.detach(), source.manifest, source.length, check_current=current)
        except Exception as exc:
            errors.append(exc)
    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    claimed = claimed or source
    handle = SourceCapability.from_fd(claimed.manifest, claimed.length, client.detach())
    try:
        yield handle, errors
    finally:
        handle.close()
        thread.join(10)
        assert not thread.is_alive(), "closed source left its native lease owner blocked"


def exercise():
    with tempfile.TemporaryDirectory(prefix="tensorfs-model-reader-") as temporary:
        store = tensorfs.Store.init(Path(temporary) / "store")
        source, transaction, data = fixture(store)
        with source_handle(store, source) as (handle, errors):
            observed = handle.inspect()
            assert tuple(observed.components) == ("zeta", "alpha")
            identity = observed.identities["zeta", "weight"]
            assert identity == observed.identities["alpha", "renamed"]
            tensorfs.gc(store.root)  # metadata-only planning must not pin a read lease
            output = bytearray(len(data))
            handle.read_part_into("zeta", "weight", "value", 0, output)
            assert output == data
            handle.check()
            assert not errors
        changed, changed_transaction, _ = fixture(store, 2, changed=True)
        with source_handle(store, changed) as (handle, errors):
            assert handle.inspect().identities["zeta", "weight"] != identity
            assert not errors
        for component, key, role, offset in [("other", "weight", "value", 0), ("zeta", "absent", "value", 0), ("zeta", "weight", "other", 0), ("zeta", "weight", "value", len(data) - 1), ("zeta", "weight", "value", -1)]:
            with source_handle(store, source) as (handle, errors):
                untouched = bytearray(b"x" * 8)
                try:
                    handle.read_part_into(component, key, role, offset, untouched)
                except tensorfs.errors.Refusal:
                    pass
                else:
                    raise AssertionError("invalid semantic part/range was accepted")
                assert untouched == b"x" * 8
            assert len(errors) == 1
        for forged in [Source("sha256:" + "ff" * 32, source.length), Source(source.manifest, source.length + 1)]:
            with source_handle(store, source, claimed=forged) as (handle, errors):
                try:
                    handle.read_part_into("zeta", "weight", "value", 0, bytearray(8))
                except tensorfs.errors.Refusal as exc:
                    assert exc.code == "TRANSACTION_CONFLICT"
                else:
                    raise AssertionError("forged source identity read native bytes")
        live = [True]
        def current():
            if not live[0]:
                raise ChannelRefusal("LEASE_REVOKED", "execution changed")
        with source_handle(store, source, current=current) as (handle, errors):
            handle.check()
            live[0] = False
            try:
                handle.read_part_into("zeta", "weight", "value", 0, bytearray(8))
            except tensorfs.errors.Refusal as exc:
                assert exc.code == "LEASE_REVOKED"
            else:
                raise AssertionError("fenced source read succeeded")
        with source_handle(store, source) as (handle, errors):
            handle.check()
            store.derived_dispose(transaction)
            try:
                tensorfs.gc(store.root)
            except tensorfs.errors.StoreBusy:
                pass
            else:
                raise AssertionError("native GC ignored an active read lease")
            output = bytearray(len(data))
            handle.read_part_into("zeta", "weight", "value", 0, output)
            assert output == data and not errors
        store.derived_dispose(changed_transaction)
        tensorfs.gc(store.root)
        try:
            store.acquire_cozytensors(source.manifest)
        except tensorfs.errors.Refusal:
            pass
        else:
            raise AssertionError("closed unretained reader prevented actual collection")
        print("PASS scoped 2MiB semantic reads, native encoded identity, malformed/forged/fenced refusal, active-lease GC custody and release", flush=True)


def isolation():
    assert os.getuid() == 0
    with tempfile.TemporaryDirectory(prefix="tensorfs-reader-uid-") as root:
        store = tensorfs.Store.init(Path(root) / "store")
        source, _, data = fixture(store)
        client, owner = socket.socketpair()
        process = subprocess.Popen([sys.executable, __file__, "--isolated-client", store.root, source.manifest, str(source.length), str(client.fileno()), hashlib.sha256(data).hexdigest()], pass_fds=(client.fileno(),))
        client.close()
        serve_source(store, owner.detach(), source.manifest, source.length)
        assert process.wait(10) == 0
        print("PASS UID65533 cannot open Store and reads 2MiB plus native geometry/identity only through its exact source FD", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--isolation"]:
        isolation()
    elif len(sys.argv) == 7 and sys.argv[1] == "--isolated-client":
        os.setgroups([]); os.setgid(65533); os.setuid(65533)
        try:
            tensorfs.Store.open(sys.argv[2])
        except tensorfs.errors.Refusal:
            pass
        else:
            raise AssertionError("isolated source reader opened Store")
        with SourceCapability.from_fd(sys.argv[3], int(sys.argv[4]), int(sys.argv[5])) as source:
            source.check()
            assert source.inspect().identities
            data = bytearray(2 << 20)
            source.read_part_into("zeta", "weight", "value", 0, data)
            assert hashlib.sha256(data).hexdigest() == sys.argv[6]
    else:
        exercise()
