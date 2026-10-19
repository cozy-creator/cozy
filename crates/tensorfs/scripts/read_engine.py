#!/usr/bin/env python3
"""Host engine contract against the installed wheel, with NumPy as a DLPack consumer.

    <wheel-python> scripts/read_engine.py
"""

from __future__ import annotations

import gc
import pathlib
import sys
import tempfile
import unittest
from collections.abc import Mapping

import numpy as np

import tensorfs as tfs


class HostReads(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.store = tfs.Store(pathlib.Path(cls.tmp.name) / "store")
        writer = cls.store.writer()
        component = writer.component("model", cls="test.Model")
        cls.plain = np.arange(12, dtype=np.float32).reshape(3, 4)
        component.put("plain", "F32", cls.plain.shape, cls.plain)
        component.put("empty", "U8", [0], b"")
        cls.big = np.arange((64 << 20) + 257, dtype=np.uint8)
        component.put("big", "U8", cls.big.shape, cls.big)
        component.put_encoded(
            "quantized", "F32", [2, 16], "fp8.rowwise",
            {"weight": ("F8_E4M3", [2, 16], bytes([0x38]) * 32),
             "scale": ("F32", [2], np.array([0.5, 2.0], dtype=np.float32))},
            {"axis": 0, "scale_dtype": "F32"},
        )
        component.put_encoded("unknown", "U8", [4], "test.unknown", {"packed": ("U8", [4], b"test")})
        cls.ck = writer.commit()
        cls.tensors = cls.ck.components["model"].tensors

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.tensors, cls.ck, cls.store, cls.big
        cls.tmp.cleanup()

    def test_mapping_and_batch_reads(self) -> None:
        tensors = self.tensors
        self.assertIsInstance(tensors, tfs.Tensors)
        self.assertIsInstance(tensors, Mapping)
        self.assertEqual(list(tensors), ["plain", "empty", "big", "quantized", "unknown"])
        self.assertEqual(tensors.keys(), self.ck.tensors["model"].keys())
        self.assertEqual(len(tensors.values()), 5)
        self.assertEqual(len(tensors.items()), 5)
        self.assertEqual(tensors["plain"].shape, (3, 4))
        self.assertIsNone(tensors.get("absent"))
        marker = object()
        self.assertIs(tensors.get("absent", marker), marker)
        with self.assertRaises(KeyError):
            tensors["absent"]
        arrays = tensors.read_all(["empty", "plain"])
        self.assertEqual(list(arrays), ["empty", "plain"])
        self.assertEqual(arrays["empty"].tobytes(), b"")
        self.assertEqual(arrays["plain"].tobytes(), self.plain.tobytes())
        self.assertEqual(list(tensors.read_all([], decode=False)), [])
        self.assertEqual(list(tensors.read_all(decode=False)), list(tensors))

    def test_dlpack_consumers_hold_the_array_storage(self) -> None:
        array = self.tensors.read("plain")
        self.assertEqual(array.__dlpack_device__(), (1, 0))
        self.assertEqual((array.dtype, array.shape, array.nbytes), ("F32", (3, 4), 48))
        first, second = np.from_dlpack(array), np.from_dlpack(array)
        copied = np.from_dlpack(array, copy=True)
        self.assertEqual(first.ctypes.data, second.ctypes.data)
        self.assertNotEqual(first.ctypes.data, copied.ctypes.data)
        self.assertEqual(first.ctypes.data % 4096, 0)
        del array
        gc.collect()
        np.testing.assert_array_equal(first, self.plain)
        np.testing.assert_array_equal(second, self.plain)
        np.testing.assert_array_equal(copied, self.plain)

    def test_writable_typed_buffers_and_chunk_spanning_offsets(self) -> None:
        destination = np.full((3, 4), -1, dtype=np.float32)
        self.assertIsNone(self.tensors.read_into("plain", destination, role=""))
        np.testing.assert_array_equal(destination, self.plain)
        out = bytearray(31)
        self.tensors.read_into("plain", memoryview(out)[3:22], offset=7)
        self.assertEqual(out, b"\0" * 3 + self.plain.tobytes()[7:26] + b"\0" * 9)
        for offset in [(64 << 20) - 17, (64 << 20) + 1]:
            count = min(200, self.big.nbytes - offset)
            out = bytearray(count)
            self.tensors.read_into("big", out, offset=offset)
            self.assertEqual(out, self.big[offset:offset + count].tobytes())
        self.assertEqual(self.tensors.read("big").tobytes(), self.big.tobytes())
        self.tensors.read_into("empty", bytearray())
        self.tensors.read_into("plain", bytearray(), offset=48)

    def test_encoded_raw_parts_and_logical_bytes(self) -> None:
        raw = self.tensors.read("quantized", decode=False)
        self.assertIsInstance(raw, tfs.Encoded)
        self.assertEqual(raw.scheme, "fp8.rowwise")
        self.assertEqual(raw.params, {"axis": 0, "scale_dtype": "F32"})
        self.assertEqual(set(raw.parts), {"weight", "scale"})
        self.assertEqual(raw.parts["weight"].tobytes(), bytes([0x38]) * 32)
        scale = bytearray(5)
        self.tensors.read_into("quantized", scale, role="scale", offset=2)
        self.assertEqual(scale, raw.parts["scale"].tobytes()[2:7])
        want = np.repeat(np.array([0.5, 2.0], dtype=np.float32), 16).reshape(2, 16)
        np.testing.assert_array_equal(np.from_dlpack(self.tensors.read("quantized")), want)
        logical = bytearray(19)
        self.tensors.read_into("quantized", logical, offset=63)
        self.assertEqual(logical, want.tobytes()[63:82])
        self.assertEqual(self.tensors.read("unknown", decode=False).parts["packed"].tobytes(), b"test")
        with self.assertRaises(tfs.UnknownSchemeError):
            self.tensors.read("unknown")

    def test_invalid_destinations_ranges_roles_and_devices(self) -> None:
        for out in [b"readonly", memoryview(bytearray(20))[::2]]:
            with self.assertRaises(tfs.InvalidError):
                self.tensors.read_into("plain", out)
        for kwargs in [{"offset": 48}, {"offset": 49}, {"offset": 2**64 - 1}, {"role": "nope"}]:
            out = bytearray(b"unchanged")
            with self.assertRaises(tfs.InvalidError):
                self.tensors.read_into("plain", out, **kwargs)
            self.assertEqual(out, b"unchanged")
        with self.assertRaises(tfs.InvalidError):
            self.tensors.read_into("quantized", bytearray(1), role="")
        with self.assertRaises(tfs.UnsupportedError):
            self.tensors.read_all(device="cuda:0")

    def test_header_metadata_does_not_imply_local_bytes(self) -> None:
        header = next((pathlib.Path(self.tmp.name) / "store").rglob(self.ck.id.removeprefix("sha256:")))
        parsed = tfs.Checkpoint.parse(header.read_bytes())
        tensors = parsed.components["model"].tensors
        self.assertEqual(list(tensors), list(self.tensors))
        for call in [lambda: tensors.read("plain"), lambda: tensors.read_all(),
                     lambda: tensors.read_into("plain", bytearray(4))]:
            with self.assertRaises(tfs.UnsupportedError):
                call()


if __name__ == "__main__":
    checks = unittest.main(verbosity=2, exit=False)
    assert "torch" not in sys.modules
    raise SystemExit(not checks.result.wasSuccessful())
