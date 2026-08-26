#!/usr/bin/env python3
"""Plant one defect in a copy of a real GGUF metadata region (tfs-011 red arms).

Every plant is a SAME-LENGTH byte patch of a copy: the file's layout is untouched, so what
the planner sees is exactly the defect and nothing else. The banked probes in
`~/cozy_v2/tracker-v2/h3-evidence` are read-only evidence and are never edited — this reads one and
writes a mutated copy somewhere else.

  scripts/gguf-plant.py <in> <out> <plant>

  no-arch          rename `general.architecture` so no architecture is declared
  mixed-arch       rename a KV into a namespace that is not the declared architecture
  no-orig-shape    rename every `comfy.gguf.orig_shape.` key so the shapes lose their source
  bad-type         set one tensor's ggml type id to a removed enumerant (5, ex-Q4_3)
  misalign         push one tensor's data offset one byte off the alignment grid
  overlap          move one tensor's data offset back behind its predecessor's end
  short-metadata   cut the probe mid tensor-info table
"""
import pathlib
import struct
import sys

PLANTS = ("no-arch", "mixed-arch", "no-orig-shape", "bad-type", "misalign", "overlap",
          "short-metadata")


class Reader:
    def __init__(self, b):
        self.b, self.o = b, 0

    def take(self, n):
        s = self.b[self.o:self.o + n]
        if len(s) != n:
            sys.exit("probe ends mid-field")
        self.o += n
        return s

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def string(self):
        n = self.u64()
        at = self.o
        return self.take(n).decode(), at, n

    def value(self, t):
        if t == 8:
            return self.string()[0]
        if t == 9:
            et, n = self.u32(), self.u64()
            return [self.value(et) for _ in range(n)]
        fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
               10: "<Q", 11: "<q", 12: "<d"}[t]
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]


def parse(b):
    """Field OFFSETS, which is all a same-length patch needs."""
    r = Reader(b)
    assert r.take(4) == b"GGUF", "not a GGUF"
    version, ntensor, nkv = r.u32(), r.u64(), r.u64()
    kv = []
    for _ in range(nkv):
        name, at, n = r.string()
        t = r.u32()
        r.value(t)
        kv.append((name, at, n))
    tensors = []
    for _ in range(ntensor):
        name, at, n = r.string()
        nd = r.u32()
        dims = [r.u64() for _ in range(nd)]
        type_at = r.o
        r.u32()
        off_at = r.o
        off = r.u64()
        tensors.append({"name": name, "dims": dims, "type_at": type_at,
                        "offset_at": off_at, "offset": off})
    return version, kv, tensors, r.o


def main():
    if len(sys.argv) != 4 or sys.argv[3] not in PLANTS:
        sys.exit(__doc__)
    src, dst, plant = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
    b = bytearray(src.read_bytes())
    _, kv, tensors, meta_end = parse(bytes(b))

    def rename(old, new, limit=None):
        assert len(old) == len(new), "a plant patches bytes, it never resizes them"
        hits = [(at, n) for name, at, n in kv if name == old or
                (limit == "prefix" and name.startswith(old))]
        if not hits:
            sys.exit(f"no key {old!r} to rename")
        for at, n in hits:
            cur = b[at:at + n].decode()
            b[at:at + n] = (new + cur[len(old):]).encode()
        return len(hits)

    if plant == "no-arch":
        n = rename("general.architecture", "general.architectvre")
        what = f"renamed {n} architecture key(s)"
    elif plant == "mixed-arch":
        n = rename("general.quantization_version", "llamacpp.quantization_versio")
        what = f"moved {n} key(s) into a foreign architecture namespace"
    elif plant == "no-orig-shape":
        n = rename("comfy.gguf.orig_shape.", "comfy.gguf.orig_shapX.", limit="prefix")
        what = f"renamed {n} orig_shape key(s)"
    elif plant == "bad-type":
        t = next(t for t in tensors if len(t["dims"]) == 2)
        b[t["type_at"]:t["type_at"] + 4] = struct.pack("<I", 5)
        what = f"{t['name']} now claims ggml type 5 (removed enumerant)"
    elif plant == "misalign":
        t = tensors[-1]
        b[t["offset_at"]:t["offset_at"] + 8] = struct.pack("<Q", t["offset"] + 1)
        what = f"{t['name']} offset {t['offset']} -> {t['offset'] + 1}"
    elif plant == "overlap":
        t = tensors[-1]
        b[t["offset_at"]:t["offset_at"] + 8] = struct.pack("<Q", tensors[-2]["offset"])
        what = f"{t['name']} offset moved onto {tensors[-2]['name']}'s"
    else:
        b = b[:meta_end - 64]
        what = "probe cut 64 B before the end of the tensor info table"
    dst.write_bytes(bytes(b))
    print(f"planted {plant}: {what} -> {dst}")


main()
