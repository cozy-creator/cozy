#!/usr/bin/env python3
"""Regenerate the shared header-only selection vectors (Rust, Go and Python consumers).
uv run --no-project --with cbor2==6.1.4 python conformance/make_select.py
"""
import hashlib
import json
from pathlib import Path
import cbor2

ROOT = Path(__file__).parent / "vectors"
HEADERS = ROOT / "select"
HEADERS.mkdir(exist_ok=True)
profiles = {}
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
    dominant = sorted(held, key=lambda k: (-held[k], k))[0]
    family, bits = {"BF16": ("bf16",16), "F16": ("fp16",16), "F32": ("f32",16),
                    "int8.rowwise": ("int8",8), "fp8.rowwise": ("fp8",8),
                    "nvfp4": ("nvfp4",4), "mxfp4": ("mxfp4",4), "int4.convrot": ("int4",4)}.get(dominant,("",0))
    profiles[name] = dict(family=family,bits=bits,bytes=sum(held.values()),kinds=sorted(held))
for name,kind,n in [("bf16","BF16",128),("fp16","F16",128),("f32","F32",256),("int8","int8.rowwise",64),
                    ("fp8","fp8.rowwise",64),("fp8-small","fp8.rowwise",32),("nvfp4","nvfp4",32),
                    ("mxfp4","mxfp4",32),("int4","int4.convrot",16),("integer","I64",128),("future","future.scheme",64)]:
    header(name,[(kind,n)])
header("future-int8",[("int8.future",64)])
header("mixed-unsupported",[("BF16",128),("future.scheme",32)])
header("mixed-fp8",[("int8.rowwise",128),("fp8.rowwise",32)])
header("tie",[("F16",128),("BF16",128)])
paths={"weights/accurate.cozytensors":"bf16","small.cozytensors":"int8","nested/fp.cozytensors":"fp8","q4.cozytensors":"nvfp4"}
cases=[]
def case(why, variants, gpus, bits, chosen, reasons=(), **extra):
    constraints={"gpus":gpus}
    if bits is not None: constraints["bits"]=bits
    cases.append(dict(why=why,variants=variants,constraints=constraints,variant=chosen,reasons=list(reasons),**extra))
case("CPU selects header bf16, not a filename",paths,[],None,"weights/accurate.cozytensors")
case("Ampere prefers int8",paths,["8.6"],None,"small.cozytensors")
case("Ada prefers int8 over fp8",paths,["8.9"],8,"small.cozytensors")
case("SM103 has no int8",paths,["10.3"],8,"nested/fp.cozytensors")
case("four bit is weight-only on Ampere",paths,["8.6"],4,"q4.cozytensors")
case("explicit sixteen",paths,["10.0"],16,"weights/accurate.cozytensors")
case("intersection of mixed GPUs",paths,["8.6","8.9"],8,"small.cozytensors")
case("mixed SM103 removes int8",paths,["10.0","10.3"],8,"nested/fp.cozytensors")
case("unknown capability is conservative",paths,[""],8,"weights/accurate.cozytensors")
case("a misleading name carries no precision",{"int8.cozytensors":"bf16","bf16.cozytensors":"int8"},["9.0"],8,"bf16.cozytensors")
case("equal family chooses smaller stored bytes",{"a.cozytensors":"fp8","z.cozytensors":"fp8-small"},["9.0"],8,"z.cozytensors",["precision 8: acme/model@1 has no int8 lane for sm90; served fp8"])
case("equal profiles choose lexicographic path",{"z.cozytensors":"int8","a.cozytensors":"int8"},["9.0"],8,"a.cozytensors")
case("native FP4 precedes smaller weight-only int4",{"a.cozytensors":"int4","z.cozytensors":"nvfp4"},["10.0"],4,"z.cozytensors")
case("all weight-only ranks by bytes",{"a.cozytensors":"nvfp4","z.cozytensors":"int4"},["8.6"],4,"z.cozytensors")
case("fallback emits a reason",{"plain.cozytensors":"bf16"},["9.0"],8,"plain.cozytensors",["precision 8: acme/model@1 has no int8 or fp8 lane for sm90; served bf16"])
case("Turing prefers fp16",{"a.cozytensors":"bf16","b.cozytensors":"fp16"},["7.5"],8,"b.cozytensors",["precision 8: sm75 runs no bf16; served fp16"])
case("F32 last-resort precision family",{"a.cozytensors":"f32","z.cozytensors":"future"},[],8,"a.cozytensors")
case("plain integer-only single variant",{"index.cozytensors":"integer"},[],8,"index.cozytensors")
case("unknown encoding is not rescued by sole variant",{"future.cozytensors":"future"},["10.0"],8,None)
case("known GPU-incompatible sole variant is skipped",{"int8.cozytensors":"int8"},[],8,None)
case("all kinds must run",{"a.cozytensors":"mixed-unsupported","b.cozytensors":"bf16"},[],8,"b.cozytensors")
case("a minority fp8 leaf still needs fp8",{"a.cozytensors":"mixed-fp8","b.cozytensors":"bf16"},["8.6"],8,"b.cozytensors",["precision 8: acme/model@1 has no int8 lane for sm86; served bf16"])
case("dominant-kind tie uses lexical kind",{"tie.cozytensors":"tie"},[],8,"tie.cozytensors")
case("no variants",{},["9.0"],8,None)
case("sixteen never falls to a sole encoded profile",{"q.cozytensors":"int8"},["9.0"],16,None)
case("sixteen excludes minority encoded leaves",{"q.cozytensors":"mixed-fp8"},["9.0"],16,None)
case("overflow capability is unknown",paths,["4294967295.0"],8,"weights/accurate.cozytensors")
case("unknown scheme inside known family is skipped",{"q.cozytensors":"future-int8"},["9.0"],8,None)
case("invalid bits",paths,["9.0"],3,None,error="invalid")
lookups=[
 {"paths":["nested/.cozytensors"],"ask":"nested/.cozytensors","variant":"nested/.cozytensors"},
 {"paths":["bf16.cozytensors","model.bf16.cozytensors"],"ask":"bf16","variant":"bf16.cozytensors"},
 {"paths":["model.bf16.cozytensors"],"ask":"bf16","variant":"model.bf16.cozytensors"},
 {"paths":["nested/model.bf16.cozytensors","other/model.bf16.cozytensors"],"ask":"bf16","variant":None},
 {"paths":["nested/anything.cozytensors"],"ask":"nested/anything","variant":"nested/anything.cozytensors"},
 {"paths":["weights.cozytensors","model.weights.cozytensors"],"ask":"model.weights.cozytensors","variant":"model.weights.cozytensors"},
]
(ROOT/"select.json").write_text(json.dumps(dict(model="acme/model@1",profiles=profiles,cases=cases,lookups=lookups),indent=2)+"\n")
print(f"{len(profiles)} headers, {len(cases)} selection cases, {len(lookups)} lookup cases")
