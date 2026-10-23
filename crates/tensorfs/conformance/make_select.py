#!/usr/bin/env python3
"""Regenerate the shared header-only selection vectors (Rust, Go and Python consumers).
uv run --no-project --with numpy --with cbor2==6.1.4 python -I conformance/make_select.py
"""
import hashlib
import json
import sys
from pathlib import Path
import cbor2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schemes import SCHEMES  # noqa: E402  every scheme a reader implements (spec §6)

ROOT = Path(__file__).parent / "vectors"
HEADERS = ROOT / "select"
HEADERS.mkdir(exist_ok=True)
TABLE = {"BF16": ("bf16", 16), "F16": ("fp16", 16), "F32": ("f32", 16), "nvfp4": ("nvfp4", 4), "mxfp4": ("mxfp4", 4),
         "w4a8.convrot": ("w4a8", 4)}
profiles = {}


def family(kind):
    for prefix, row in (("int8.", ("int8", 8)), ("fp8.", ("fp8", 8)), ("int4.", ("int4", 4))):
        if kind.startswith(prefix):
            return row
    return TABLE.get(kind, ("", 0))


def header(name, entries):
    encodings, tensors, offset = [], [], 0
    held = {}
    for i, (kind, length) in enumerate(entries):
        if kind.isupper():
            width = {"BF16": 2, "F16": 2, "F32": 4, "I64": 8}.get(kind, 1)
            tensors.append([str(i), kind, [length // width], 0, [0, offset, length]])
        else:
            encodings.append([kind, {}])
            tensors.append([str(i), "BF16", [length], len(encodings), [["weight", "U8", [length], [0, offset, length]]]])
        held[kind] = held.get(kind, 0) + length
        offset += ((length + 4095) // 4096) * 4096
    raw = ["cozytensors/1", [[hashlib.sha256(name.encode()).digest(), offset]], encodings, [],
           [["model", ["", "", ""], {}, tensors, {}]], {}]
    (HEADERS / f"{name}.cbor").write_bytes(cbor2.dumps(raw, canonical=True))
    # Its lane: the table kind with the fewest bits; of equal bits, the most bytes, then the name.
    # An encoded kind the table does not name leaves it no lane.
    lanes = [] if any(not k.isupper() and not family(k)[1] for k in held) else [k for k in held if family(k)[1]]
    fam, bits = family(min(lanes, key=lambda k: (family(k)[1], -held[k], k))) if lanes else ("", 0)
    profiles[name] = dict(family=fam, bits=bits, bytes=sum(held.values()), kinds=sorted(held))


for name, kind, n in [("bf16", "BF16", 128), ("fp16", "F16", 128), ("f32", "F32", 256), ("int8", "int8.rowwise", 64),
                      ("fp8", "fp8.rowwise", 64), ("fp8-small", "fp8.rowwise", 32), ("nvfp4", "nvfp4", 32),
                      ("mxfp4", "mxfp4", 32), ("int4", "int4.convrot", 16), ("integer", "I64", 128), ("future", "future.scheme", 64)]:
    header(name, [(kind, n)])
header("future-int8", [("int8.future", 64)])
header("mixed-unsupported", [("BF16", 128), ("future.scheme", 32)])
header("mixed-fp8", [("int8.rowwise", 128), ("fp8.rowwise", 32)])
header("tie", [("F16", 128), ("BF16", 128)])
header("integer-mixed-int8", [("I64", 128), ("int8.rowwise", 32)])
# H3's layout: its nvfp4 variant keeps sensitive layers in int8, which hold most of its bytes.
header("int8-convrot", [("int8.convrot", 160), ("BF16", 16)])
header("nvfp4-int8-heavy", [("int8.convrot", 96), ("nvfp4", 32), ("BF16", 16)])
# A w4a8 variant (xs-062): the blocks in w4a8.convrot, the text encoder int8, as a lane recipe writes it; and
# kitchen's own scheme beside int8, which has no lane (a reader of rule 3 read it as int8).
header("w4a8-int8", [("w4a8.convrot", 36), ("int8.convrot", 40), ("BF16", 16)])
header("kitchen-int8", [("comfy.asym_w4a8_int8", 36), ("int8.convrot", 40), ("BF16", 16)])
for scheme in sorted(SCHEMES):
    header(f"scheme-{scheme}", [(scheme, 64)])

M = "acme/model@1"
paths = {"weights/accurate.cozytensors": "bf16", "small.cozytensors": "int8", "nested/fp.cozytensors": "fp8", "q4.cozytensors": "nvfp4"}
cases = []


def case(why, variants, gpus, chosen, reasons=(), refused=None, error=None, **constraints):
    row = dict(why=why, variants=variants, constraints=dict(gpus=gpus, **constraints), variant=chosen, reasons=list(reasons))
    for key, value in (("refused", refused), ("error", error)):
        if value is not None:
            row[key] = value
    cases.append(row)


def none(bits, gpus, *published):
    """The refusal when no variant works: what each is and why (spec §7)."""
    return f"precision {bits} on {gpus}: {M} has no variant that works; it publishes {'; '.join(published) or 'none'}"


# No bits: the lane order is a preference; any variant that works serves when none in it does.
case("CPU selects header bf16, not a filename", paths, [], "weights/accurate.cozytensors")
case("Ampere prefers int8", paths, ["8.6"], "small.cozytensors")
case("a mixed release on Ada takes int8", paths, ["8.9"], "small.cozytensors")
case("a mixed release on Turing takes bf16", paths, ["7.5"], "weights/accurate.cozytensors",
     ["precision 8: acme/model@1 has no fp16 lane for sm75; served bf16"])
case("unknown capability is conservative", paths, [""], "weights/accurate.cozytensors")
case("overflow capability is unknown", paths, ["4294967295.0"], "weights/accurate.cozytensors")
case("fallback emits a reason", {"plain.cozytensors": "bf16"}, ["9.0"], "plain.cozytensors",
     ["precision 8: acme/model@1 has no int8 or fp8 lane for sm90; served bf16"])
case("Turing prefers fp16", {"a.cozytensors": "bf16", "b.cozytensors": "fp16"}, ["7.5"], "b.cozytensors",
     ["precision 8: sm75 runs no bf16; served fp16"])
case("F32 last-resort precision family", {"a.cozytensors": "f32", "z.cozytensors": "future"}, [], "a.cozytensors")
case("plain integer-only single variant", {"index.cozytensors": "integer"}, [], "index.cozytensors")
case("dominant-kind tie uses lexical kind", {"tie.cozytensors": "tie"}, [], "tie.cozytensors")
case("all kinds must run on a GPU", {"a.cozytensors": "mixed-unsupported", "b.cozytensors": "bf16"}, ["9.0"], "b.cozytensors",
     ["precision 8: acme/model@1 has no int8 or fp8 lane for sm90; served bf16"])
case("a minority fp8 leaf still needs fp8", {"a.cozytensors": "mixed-fp8", "b.cozytensors": "bf16"}, ["8.6"], "b.cozytensors",
     ["precision 8: acme/model@1 has no int8 lane for sm86; served bf16"])
case("a sole 8-bit variant serves the CPU", {"model.cozytensors": "int8"}, [], "model.cozytensors",
     ["precision 8: acme/model@1 has no bf16 or fp16 lane for no GPU; served int8"])
case("a sole fp8 variant serves the CPU", {"model.cozytensors": "fp8"}, [], "model.cozytensors",
     ["precision 8: acme/model@1 has no bf16 or fp16 lane for no GPU; served fp8"])
case("a sole 8-bit variant serves a GPU that runs it", {"model.cozytensors": "int8"}, ["8.9"], "model.cozytensors")
case("a sole 8-bit variant is refused on a GPU that runs none of it", {"model.cozytensors": "int8"}, ["7.5"], None,
     refused=none(8, "sm75", "model.cozytensors (int8) needs sm80+ except sm103 for int8.rowwise"))
case("one GPU of a set running it is enough", {"model.cozytensors": "fp8"}, ["8.6", "8.9"], "model.cozytensors",
     ["precision 8: acme/model@1 has no int8, bf16 or fp16 lane for sm86+sm89; served fp8"])
case("an unknown encoding is refused on a GPU", {"future.cozytensors": "future"}, ["10.0"], None,
     refused=none(8, "sm100", "future.cozytensors (future.scheme) needs a GPU reader for future.scheme"))
case("an unknown scheme inside a known family is refused on a GPU", {"q.cozytensors": "future-int8"}, ["9.0"], None,
     refused=none(8, "sm90", "q.cozytensors (int8) needs a GPU reader for int8.future"))
case("no variants", {}, ["9.0"], None, refused=none(8, "sm90"))
case("a scheme outside the table decodes on any GPU", {"q.cozytensors": "scheme-gguf.q4_0"}, ["9.0"], "q.cozytensors")
# A tableless encoding beside plain weights has no lane, so it is never read as bf16 (or as int8 beside int8 extras:
# how a reader of rule 3 misread w4a8); it serves only when no lane variant works.
header("gguf-bf16", [("gguf.q4_0", 64), ("BF16", 128)])
case("a tableless encoding beside bf16 has no lane", {"a.cozytensors": "gguf-bf16", "b.cozytensors": "bf16"}, ["9.0"],
     "b.cozytensors", ["precision 8: acme/model@1 has no int8 or fp8 lane for sm90; served bf16"])
case("a tableless encoding serves when no lane variant works", {"a.cozytensors": "gguf-bf16", "b.cozytensors": "int8"}, ["7.5"],
     "a.cozytensors")
# Explicit bits: only that lane answers.
case("Ada prefers int8 over fp8", paths, ["8.9"], "small.cozytensors", bits=8)
case("SM103 has no int8", paths, ["10.3"], "nested/fp.cozytensors", bits=8)
case("four bit is weight-only on Ampere", paths, ["8.6"], "q4.cozytensors",
     ["precision 4: acme/model@1 has no w4a8 lane for sm86; served nvfp4"], bits=4)
case("explicit sixteen", paths, ["10.0"], "weights/accurate.cozytensors", bits=16)
case("intersection of mixed GPUs", paths, ["8.6", "8.9"], "small.cozytensors", bits=8)
case("mixed SM103 removes int8", paths, ["10.0", "10.3"], "nested/fp.cozytensors", bits=8)
case("a misleading name carries no precision", {"int8.cozytensors": "bf16", "bf16.cozytensors": "int8"}, ["9.0"], "bf16.cozytensors", bits=8)
case("equal family chooses smaller stored bytes", {"a.cozytensors": "fp8", "z.cozytensors": "fp8-small"}, ["9.0"], "z.cozytensors",
     ["precision 8: acme/model@1 has no int8 lane for sm90; served fp8"], bits=8)
case("equal profiles choose lexicographic path", {"z.cozytensors": "int8", "a.cozytensors": "int8"}, ["9.0"], "a.cozytensors", bits=8)
case("native FP4 precedes smaller weight-only int4", {"a.cozytensors": "int4", "z.cozytensors": "nvfp4"}, ["10.0"], "z.cozytensors", bits=4)
case("all weight-only ranks by bytes", {"a.cozytensors": "nvfp4", "z.cozytensors": "int4"}, ["8.6"], "z.cozytensors", bits=4)
case("a sole 8-bit variant serves the CPU at bits 8", {"model.cozytensors": "int8"}, [], "model.cozytensors",
     ["precision 8: acme/model@1 has no bf16 or fp16 lane for no GPU; served int8"], bits=8)
case("sixteen refuses a sole 8-bit variant", {"q.cozytensors": "int8"}, ["9.0"], None, bits=16,
     refused=none(16, "sm90", "q.cozytensors (int8) is not 16-bit"))
case("sixteen refuses a sole 8-bit variant on the CPU", {"q.cozytensors": "int8"}, [], None, bits=16,
     refused=none(16, "no GPU", "q.cozytensors (int8) is not 16-bit"))
case("sixteen excludes minority encoded leaves", {"q.cozytensors": "mixed-unsupported"}, ["9.0"], None, bits=16,
     refused=none(16, "sm90", "q.cozytensors (BF16+future.scheme) is not 16-bit"))
case("an integer-dominant variant holding int8 is 8-bit", {"index.cozytensors": "integer-mixed-int8"}, ["9.0"], None, bits=16,
     refused=none(16, "sm90", "index.cozytensors (int8) is not 16-bit"))
case("eight refuses a 16-bit release", {"plain.cozytensors": "bf16"}, ["9.0"], None, bits=8,
     refused=none(8, "sm90", "plain.cozytensors (bf16) is not 8-bit"))
case("four refuses an 8-bit release and names each variant", {"a.cozytensors": "int8", "b.cozytensors": "bf16"}, ["8.9"], None, bits=4,
     refused=none(4, "sm89", "a.cozytensors (int8) is not 4-bit", "b.cozytensors (bf16) is not 4-bit"))
case("invalid bits", paths, ["9.0"], None, bits=3, error="invalid")
h3 = {"model.int8.cozytensors": "int8-convrot", "model.nvfp4.cozytensors": "nvfp4-int8-heavy"}
case("a variant holding nvfp4 is 4-bit, however many int8 bytes it holds", h3, ["12.0"], "model.int8.cozytensors", bits=8)
case("the default lane takes int8 over a smaller nvfp4 variant", h3, ["12.0"], "model.int8.cozytensors")
case("four bits take the nvfp4 variant", h3, ["12.0"], "model.nvfp4.cozytensors", bits=4)
case("the nvfp4 variant serves 4 bits before Blackwell, weight-only", h3, ["8.9"], "model.nvfp4.cozytensors",
     ["precision 4: acme/model@1 has no w4a8 lane for sm89; served nvfp4"], bits=4)
# w4a8 (xs-062): 4 bits on int8 tensor cores, first before Blackwell, after nvfp4 on it.
w4 = {**h3, "model.w4a8.cozytensors": "w4a8-int8"}
case("four bits take w4a8 on Ampere", w4, ["8.6"], "model.w4a8.cozytensors", bits=4)
case("four bits take w4a8 on Hopper", w4, ["9.0"], "model.w4a8.cozytensors", bits=4)
case("four bits take nvfp4 on Blackwell", w4, ["12.0"], "model.nvfp4.cozytensors", bits=4)
case("Blackwell without nvfp4 takes w4a8", {k: v for k, v in w4.items() if "nvfp4" not in k}, ["10.0"], "model.w4a8.cozytensors",
     ["precision 4: acme/model@1 has no nvfp4 lane for sm100; served w4a8"], bits=4)
case("the default lane takes int8 over a smaller w4a8 variant", w4, ["8.6"], "model.int8.cozytensors")
kitchen = {**h3, "model.kitchen.cozytensors": "kitchen-int8"}
case("kitchen's scheme beside int8 has no lane", kitchen, ["8.6"], "model.int8.cozytensors")
case("kitchen's scheme is not the w4a8 lane", kitchen, ["8.6"], "model.nvfp4.cozytensors",
     ["precision 4: acme/model@1 has no w4a8 lane for sm86; served nvfp4"], bits=4)
case("eight never takes a w4a8 variant", w4, ["9.0"], "model.int8.cozytensors", bits=8)
case("a decode slot's own 4 bits take nvfp4 before w4a8", w4, ["8.9"], "model.nvfp4.cozytensors", prefer=4, min_bits=4)
case("a decode slot takes w4a8 without nvfp4", {k: v for k, v in w4.items() if "nvfp4" not in k}, ["8.9"],
     "model.w4a8.cozytensors", ["precision 4: acme/model@1 has no nvfp4 lane for sm89; served w4a8"], prefer=4, min_bits=4)
case("asked 4 bits on a decode slot take w4a8", w4, ["8.9"], "model.w4a8.cozytensors", bits=4, min_bits=4)
case("B300 has no int8, so no w4a8", {"model.w4a8.cozytensors": "w4a8-int8"}, ["10.3"], None, bits=4,
     refused=none(4, "sm103", "model.w4a8.cozytensors (w4a8) needs sm80+ except sm103 for int8.convrot "
                  "and sm80+ except sm103 for w4a8.convrot"))
case("the CPU takes int8 over a smaller w4a8 variant unasked", {k: v for k, v in w4.items() if "nvfp4" not in k}, [],
     "model.int8.cozytensors", ["precision 8: acme/model@1 has no bf16 or fp16 lane for no GPU; served int8"])
case("the CPU takes int8 over a smaller nvfp4 variant unasked", h3, [], "model.int8.cozytensors",
     ["precision 8: acme/model@1 has no bf16 or fp16 lane for no GPU; served int8"])
case("a sole w4a8 variant serves the CPU", {"model.w4a8.cozytensors": "w4a8-int8"}, [], "model.w4a8.cozytensors",
     ["precision 4: acme/model@1 has no bf16 or fp16 lane for no GPU; served w4a8"], bits=4)
case("a run's lane is its main model's: H3's 16-bit-only LoRA is asked nothing", {"lora.cozytensors": "bf16"}, ["12.0"],
     "lora.cozytensors", ["precision 8: acme/model@1 has no int8 or fp8 lane for sm120; served bf16"])
# A slot: its lane first, never narrower than min_bits unasked, encoded weights only when it takes them.
case("a slot refusing encoded weights refuses a sole 8-bit variant", {"m.cozytensors": "int8"}, ["8.9"], None, plain=True,
     refused=none(8, "sm89", "m.cozytensors (int8) holds encoded int8.rowwise"))
case("a slot refusing encoded weights takes bf16", paths, ["9.0"], "weights/accurate.cozytensors",
     ["precision 8: acme/model@1 declines encoded leaves; served bf16"], plain=True)
case("min_bits refuses a narrower variant unasked", {"m.cozytensors": "nvfp4"}, ["9.0"], None, prefer=8, min_bits=8,
     refused=none(8, "sm90", "m.cozytensors (nvfp4) is 4-bit, under 8"))
case("asked bits outrank min_bits", {"m.cozytensors": "nvfp4"}, ["9.0"], "m.cozytensors",
     ["precision 4: acme/model@1 has no w4a8 lane for sm90; served nvfp4"], bits=4, min_bits=8)
case("a decode slot prefers 4 bits", paths, ["8.9"], "q4.cozytensors", prefer=4, min_bits=4)
case("a 16-bit slot prefers bf16 and refuses 8-bit unasked", {"a.cozytensors": "int8", "b.cozytensors": "bf16"}, ["8.9"],
     "b.cozytensors", prefer=16, min_bits=16)
# Every scheme a reader implements: one of the table runs on sm100 at its width, any other on no GPU.
for scheme in sorted(SCHEMES):
    fam, width = family(scheme)
    reasons = {"fp8": ["precision 8: acme/model@1 has no int8 lane for sm100; served fp8"],
               "w4a8": ["precision 4: acme/model@1 has no nvfp4 lane for sm100; served w4a8"]}.get(fam, [])
    chosen, reasons = ("s.cozytensors", reasons) if fam else ("f32.cozytensors", [])
    case(f"scheme {scheme}", {"s.cozytensors": f"scheme-{scheme}", "f32.cozytensors": "f32"}, ["10.0"], chosen, reasons,
         prefer=4 if width == 4 else 8)

# A variant is its exact path: no lane word or extension-less name is matched to one (spec §3).
lookups = [
    {"paths": ["nested/.cozytensors"], "ask": "nested/.cozytensors", "variant": "nested/.cozytensors"},
    {"paths": ["model.bf16.cozytensors"], "ask": "model.bf16.cozytensors", "variant": "model.bf16.cozytensors"},
    {"paths": ["bf16.cozytensors", "model.bf16.cozytensors"], "ask": "bf16", "variant": None},
    {"paths": ["model.bf16.cozytensors"], "ask": "bf16", "variant": None},
    {"paths": ["nested/anything.cozytensors"], "ask": "nested/anything", "variant": None},
    {"paths": ["model.cozytensors"], "ask": "", "variant": None},
    {"paths": ["README.md"], "ask": "README.md", "variant": None},
]
spec = dict(model=M, schemes=sorted(SCHEMES), profiles=profiles, cases=cases, lookups=lookups)
(ROOT / "select.json").write_text(json.dumps(spec, indent=2) + "\n")
print(f"{len(profiles)} headers, {len(cases)} selection cases, {len(lookups)} lookup cases")
