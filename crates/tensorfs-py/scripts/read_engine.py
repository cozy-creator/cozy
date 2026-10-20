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
from tensorfs import raw as raw_format


class HostReads(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.cache = pathlib.Path(cls.tmp.name) / "store"
        writer = tfs.Writer(cache_dir=cls.cache)
        component = writer.component("model", cls="test.Model", config={})
        cls.plain = np.arange(12, dtype=np.float32).reshape(3, 4)
        component.put("plain", cls.plain)
        component.put("empty", b"", dtype="U8", shape=[0])
        cls.big = np.arange((64 << 20) + 257, dtype=np.uint8)
        component.put("big", cls.big)
        component.put_encoded(
            "quantized", dtype="F32", shape=[2, 16], scheme="fp8.rowwise",
            parts={"weight": ("F8_E4M3", [2, 16], bytes([0x38]) * 32),
             "scale": ("F32", [2], np.array([0.5, 2.0], dtype=np.float32))},
            params={"axis": 0, "scale_dtype": "F32"},
        )
        component.put_encoded("unknown", dtype="U8", shape=[4], scheme="test.unknown", parts={"packed": ("U8", [4], b"test")}, params={})
        cls.ck = writer.commit()
        cls.tensors = cls.ck.components["model"].tensors

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.tensors, cls.ck, cls.cache, cls.big
        cls.tmp.cleanup()

    def test_mapping_and_batch_reads(self) -> None:
        tensors = self.tensors
        self.assertIsInstance(tensors, tfs.Tensors)
        self.assertIsInstance(tensors, Mapping)
        self.assertEqual(list(tensors), ["plain", "empty", "big", "quantized", "unknown"])
        self.assertEqual(list(tensors.keys()), ["plain", "empty", "big", "quantized", "unknown"])
        self.assertEqual(len(tensors.values()), 5)
        self.assertEqual(len(tensors.items()), 5)
        self.assertEqual(tensors["plain"].shape, (3, 4))
        self.assertIsInstance(tensors["plain"], tfs.TensorMetadata)
        self.assertFalse(hasattr(tensors["plain"], "parts"))
        self.assertFalse(hasattr(tensors, "read_all"))
        self.assertFalse(hasattr(tfs, "Info"))
        self.assertFalse(hasattr(tfs, "Part"))
        self.assertIsNone(tensors.get("absent"))
        marker = object()
        self.assertIs(tensors.get("absent", marker), marker)
        with self.assertRaises(KeyError):
            tensors["absent"]
        arrays = tensors.read(["empty", "plain", "empty"])
        self.assertEqual(list(arrays), ["empty", "plain"])
        self.assertEqual(arrays["empty"].tobytes(), b"")
        self.assertEqual(arrays["plain"].tobytes(), self.plain.tobytes())
        self.assertEqual(list(tensors.read([], decode=False)), [])
        self.assertEqual(list(tensors.read(decode=False)), list(tensors))

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

    def test_dlpack_refuses_dimensions_outside_its_signed_shape_abi(self) -> None:
        writer = tfs.Writer(cache_dir=self.cache)
        writer.component("empty", cls="test.Empty", config={}).put("large-shape", b"", dtype="U8", shape=[0, 2**63])
        array = writer.commit().components["empty"].tensors.read("large-shape")
        self.assertEqual(array.nbytes, 0)
        with self.assertRaises(tfs.InvalidError):
            array.__dlpack__()

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
        self.assertIsInstance(self.tensors.read("quantized"), tfs.Encoded)
        self.assertIsInstance(self.tensors.read(["quantized"])["quantized"], tfs.Encoded)
        self.assertEqual(raw.scheme, "fp8.rowwise")
        self.assertEqual(raw.params, {"axis": 0, "scale_dtype": "F32"})
        self.assertEqual(set(raw.parts), {"weight", "scale"})
        self.assertEqual(raw.parts["weight"].tobytes(), bytes([0x38]) * 32)
        scale = bytearray(5)
        self.tensors.read_into("quantized", scale, role="scale", offset=2)
        self.assertEqual(scale, raw.parts["scale"].tobytes()[2:7])
        want = np.repeat(np.array([0.5, 2.0], dtype=np.float32), 16).reshape(2, 16)
        np.testing.assert_array_equal(np.from_dlpack(self.tensors.read("quantized", decode=True)), want)
        logical = bytearray(19)
        self.tensors.read_into("quantized", logical, offset=63)
        self.assertEqual(logical, want.tobytes()[63:82])
        self.assertEqual(self.tensors.read("unknown", decode=False).parts["packed"].tobytes(), b"test")
        with self.assertRaises(tfs.UnknownSchemeError):
            self.tensors.read("unknown", decode=True)

    def test_invalid_destinations_ranges_roles_and_devices(self) -> None:
        for out in [b"readonly", memoryview(bytearray(20))[::2]]:
            with self.assertRaises(tfs.InvalidError):
                self.tensors.read_into("plain", out)
        for kwargs in [{"offset": 48}, {"offset": 49}, {"offset": 2**64 - 1}]:
            out = bytearray(b"unchanged")
            with self.assertRaises(tfs.InvalidError):
                self.tensors.read_into("plain", out, **kwargs)
            self.assertEqual(out, b"unchanged")
        for key, role in [("plain", "nope"), ("quantized", "")]:
            out = bytearray(b"unchanged")
            with self.assertRaises(tfs.NotFoundError):
                self.tensors.read_into(key, out, role=role)
            self.assertEqual(out, b"unchanged")
        for device in ["gpu", "cuda:-1", "cuda:nope"]:
            with self.assertRaises(tfs.InvalidError):
                self.tensors.read(device=device)

    def test_real_endpoint_fetches_only_chunks_touched_by_the_read(self) -> None:
        root = pathlib.Path(self.tmp.name)
        endpoint = (root / "origin").as_uri()
        tfs.materialize(self.ck, local_dir=root / "portable-source")
        tfs.upload_folder(repo_id="acme/reads", revision="one", folder_path=root / "portable-source",
                          endpoint=endpoint, cache_dir=root / "upload")
        remote = tfs.open("acme/reads", revision="one", endpoint=endpoint, cache_dir=root / "reader")
        before = set(remote.missing())
        self.assertEqual(remote.components["model"].tensors.read([]), {})
        with self.assertRaises(KeyError):
            remote.components["model"].tensors.read(["plain", "absent"])
        self.assertEqual(set(remote.missing()), before)
        part = raw_format.parts(remote, "model", "big")[""]
        chunk = remote.chunks[part.chunk]
        self.assertIn(chunk, before)
        out = bytearray(17)
        remote.components["model"].tensors.read_into("big", out, offset=1)
        self.assertEqual(out, self.big[1:18].tobytes())
        self.assertEqual(set(remote.missing()), before - {chunk})

    def test_header_metadata_does_not_imply_local_bytes(self) -> None:
        header = next((pathlib.Path(self.tmp.name) / "store").rglob(self.ck.id.removeprefix("sha256:")))
        parsed = tfs.Checkpoint.parse(header.read_bytes())
        tensors = parsed.components["model"].tensors
        self.assertEqual(tensors.read([]), {})
        with self.assertRaises(KeyError):
            tensors.read(["plain", "absent"])
        self.assertEqual(list(tensors), list(self.tensors))
        for call in [lambda: tensors.read("plain"), lambda: tensors.read(),
                     lambda: tensors.read_into("plain", bytearray(4))]:
            with self.assertRaises(tfs.UnsupportedError):
                call()


if __name__ == "__main__":
    checks = unittest.main(verbosity=2, exit=False)
    assert "torch" not in sys.modules
    raise SystemExit(not checks.result.wasSuccessful())
