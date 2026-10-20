"""Actual CPU DLPack ownership proof; run against the matching installed native wheel.

The ctypes fixture owns a real legacy DLManagedTensor allocation until its deleter.
No Torch, CUDA, model download or mock storage is involved.
"""

from __future__ import annotations

import ctypes as C
import gc
import hashlib
import json
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import tensorfs
from tensorfs import _ext


class Device(C.Structure):
    _fields_ = [("device_type", C.c_int32), ("device_id", C.c_int32)]


class Dtype(C.Structure):
    _fields_ = [("code", C.c_uint8), ("bits", C.c_uint8), ("lanes", C.c_uint16)]


class Tensor(C.Structure):
    _fields_ = [
        ("data", C.c_void_p), ("device", Device), ("ndim", C.c_int32),
        ("dtype", Dtype), ("shape", C.POINTER(C.c_int64)),
        ("strides", C.POINTER(C.c_int64)), ("byte_offset", C.c_uint64),
    ]


class Managed(C.Structure):
    pass


ManagedDeleter = C.CFUNCTYPE(None, C.POINTER(Managed))
Managed._fields_ = [("dl_tensor", Tensor), ("manager_ctx", C.c_void_p), ("deleter", ManagedDeleter)]

_live: dict[int, OwnedTensor] = {}
_callback_errors: list[str] = []
_main_thread = threading.get_ident()
_dltensor = b"dltensor"
_used = b"used_dltensor"

_new = C.pythonapi.PyCapsule_New
_new.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p]
_new.restype = C.py_object
_valid = C.pythonapi.PyCapsule_IsValid
_valid.argtypes = [C.c_void_p, C.c_char_p]
_valid.restype = C.c_int
_pointer = C.pythonapi.PyCapsule_GetPointer
_pointer.argtypes = [C.c_void_p, C.c_char_p]
_pointer.restype = C.c_void_p
_name = C.pythonapi.PyCapsule_GetName
_name.argtypes = [C.c_void_p]
_name.restype = C.c_char_p


@ManagedDeleter
def managed_delete(pointer):
    """Release the actual allocation once; callbacks stay alive at module scope."""
    address = C.addressof(pointer.contents)
    owner = _live.pop(address, None)
    if owner is None:
        _callback_errors.append("managed tensor deleter was called twice")
        return
    owner.deletions += 1
    owner.release_thread = threading.get_ident()
    owner.released_bytes = bytes(owner.allocation)
    try:
        if owner.on_delete is not None:
            owner.on_delete()
    except BaseException as error:  # noqa: BLE001 - record every failure at the C callback boundary
        _callback_errors.append(repr(error))
    finally:
        # The retained capsule now contains a genuinely stale pointer. Every reuse
        # assertion below must reject its used name before dereferencing that pointer.
        owner.allocation = None
        owner.shape = None
        owner.managed = None
        owner.released.set()


@C.CFUNCTYPE(None, C.c_void_p)
def capsule_delete(capsule):
    # Use the raw PyObject pointer: converting a dying capsule to py_object here
    # would create another Python reference during its destructor.
    if _valid(capsule, _dltensor):
        managed_delete(C.cast(_pointer(capsule, _dltensor), C.POINTER(Managed)))


class OwnedTensor:
    def __init__(self, data: bytes, on_delete=None):
        self.allocation = (C.c_uint8 * len(data)).from_buffer_copy(data)
        self.shape = (C.c_int64 * 1)(len(data))
        self.managed = Managed(
            Tensor(C.addressof(self.allocation), Device(1, 0), 1, Dtype(1, 8, 1),
                   self.shape, None, 0),
            None, managed_delete,
        )
        address = C.addressof(self.managed)
        _live[address] = self
        self.capsule = _new(address, _dltensor, C.cast(capsule_delete, C.c_void_p))
        self.deletions = 0
        self.release_thread = None
        self.released_bytes = None
        self.released = threading.Event()
        self.on_delete = on_delete

    def __dlpack_device__(self):
        return (1, 0)

    def __dlpack__(self, *, stream=None):
        assert stream is None
        return self.capsule


class CapsuleAlias:
    def __init__(self, capsule, device):
        self.capsule = capsule
        self.device = device

    def __dlpack_device__(self):
        return self.device

    def __dlpack__(self, *, stream=None):
        assert stream is None
        return self.capsule


def rejected(action):
    try:
        action()
    except tensorfs.InvalidError:
        return
    raise AssertionError("the aliased/invalid DLPack operation was accepted")


def released(owner):
    assert owner.released.wait(5), "managed owner was not released after the operation"
    assert owner.deletions == 1
    assert owner.allocation is None and owner.managed is None
    assert owner.release_thread != _main_thread, "foreign deleter bypassed cleanup worker"
    assert _name(id(owner.capsule)) == _used
    assert not _callback_errors, _callback_errors


def main():
    payload = bytes(range(64))
    passed = []
    with tempfile.TemporaryDirectory(prefix="tensorfs-dlpack-owner-") as scratch:
        tensorfs.configure(cache_dir=Path(scratch) / "store")
        writer = tensorfs.Writer()
        component = writer.component("model", cls="proof.DLPack", config={})

        def reenter_writer():
            assert component.has("source"), "deleter ran before put completed"

        source = OwnedTensor(payload, reenter_writer)
        component.put("source", source)
        assert _name(id(source.capsule)) == _used
        released(source)
        # The underlying allocation/managed struct are already freed. A second
        # claim must fail at capsule-name validation, before metadata is read.
        rejected(lambda: component.put("reuse", source))
        assert source.deletions == 1
        passed.append("writer_consumes_alias_and_reentrant_deleter_once")

        validation = OwnedTensor(payload)
        rejected(lambda: component.put("invalid", validation, dtype="F32"))
        assert _name(id(validation.capsule)) == _dltensor
        assert validation.deletions == 0 and validation.managed is not None
        component.put("validated", validation)
        released(validation)
        passed.append("validation_precedes_transfer_and_can_retry")

        values = np.arange(24, dtype=np.uint16).reshape(4, 6)
        capsule = values.__dlpack__()
        numpy_alias = CapsuleAlias(capsule, values.__dlpack_device__())
        component.put("numpy", numpy_alias)
        assert _name(id(capsule)) == _used
        rejected(lambda: component.put("numpy_reuse", numpy_alias))
        passed.append("real_numpy_capsule_consumed_once")

        checkpoint = writer.commit()
        tensors = checkpoint.components["model"].tensors
        assert tensors.read("source").tobytes() == payload
        assert tensors.read("validated").tobytes() == payload
        assert tensors.read("numpy").tobytes() == values.tobytes()
        passed.append("stored_bytes_match_independent_inputs")

        destination = OwnedTensor(bytes(64))
        tensors.read_into("source", destination)
        released(destination)
        assert destination.released_bytes == payload
        rejected(lambda: tensors.read_into("source", destination))
        assert destination.deletions == 1
        passed.append("read_into_consumes_alias_and_deleter_once")

        failed_destination = OwnedTensor(bytes(64))
        rejected(lambda: tensors.read_into("source", failed_destination, offset=1))
        released(failed_destination)
        assert failed_destination.released_bytes == bytes(64)
        passed.append("read_into_error_releases_once_without_write")

        abandoned = tensorfs.Writer()
        closed_component = abandoned.component("closed", cls="proof.DLPack", config={})
        abandoned.abandon()

        def reenter_closed_writer():
            rejected(lambda: closed_component.has("unused"))

        failed_source = OwnedTensor(payload, reenter_closed_writer)
        rejected(lambda: closed_component.put("unused", failed_source))
        released(failed_source)
        assert failed_source.released_bytes == payload
        passed.append("never_read_writer_error_releases_once_outside_mutex")

        checkpoint.close()
        gc.collect()
        time.sleep(0.02)
        assert not _live, "a managed producer allocation remains live"
        assert not _callback_errors, _callback_errors
        assert all(owner.deletions == 1 for owner in
                   [source, validation, destination, failed_destination, failed_source])
        print(json.dumps({
            "passed": len(passed), "failed": 0, "skipped": 0, "cases": passed,
            "managed_deletions": 5, "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "numpy_version": np.__version__, "tensorfs_version": tensorfs.__version__,
            "native_path": _ext.__file__,
            "native_sha256": hashlib.sha256(Path(_ext.__file__).read_bytes()).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "CPU legacy DLPack; no CUDA/stream-failure qualification",
        }, sort_keys=True))


if __name__ == "__main__":
    main()
