"""The public Python report against an actual local checkpoint; no network or repair."""

import json
import sys
from pathlib import Path

import tensorfs

request = json.load(sys.stdin)
folder = Path(request["folder"])
model, other = map(tuple, request["chunks"])
asset = tuple(request["asset"])
unused = tuple(request["unused"])
header = tuple(request["header"])
checkpoint = tensorfs.open(folder, endpoint="http://127.0.0.1:1", cache_dir=folder.parent / "python-cache")

report = checkpoint.verify()
assert isinstance(report, tensorfs.VerificationReport)
assert report.ok and report.missing == [] and report.corrupt == []
try:
    report.missing = []
except AttributeError:
    pass
else:
    raise AssertionError("verification report fields are writable")

unused_path = folder / "chunks" / unused[0]
unused_bytes = unused_path.read_bytes()
unused_path.unlink()
assert checkpoint.verify().missing == [unused]
assert checkpoint.verify(["model", "other", "shared"]).ok
unused_path.write_bytes(unused_bytes)

(folder / "chunks" / model[0]).unlink()
(folder / "chunks" / other[0]).write_bytes(b"bad-hash")
report = checkpoint.verify()
assert not report.ok and report.missing == [model] and report.corrupt == [other]
selected = checkpoint.verify(components=["model", "model"])
assert selected.missing == [model] and selected.corrupt == []
assert checkpoint.verify(["other"]).corrupt == [other]
assert not (folder / "chunks" / model[0]).exists()
assert (folder / "chunks" / other[0]).read_bytes() == b"bad-hash"
(folder / "chunks" / asset[0]).write_bytes(b"short")
assert checkpoint.verify(["model"]).corrupt == [asset]
assert checkpoint.verify([]).ok

header_path = folder / "model.cozytensors"
original_header = header_path.read_bytes()
header_path.write_bytes(bytes([original_header[0] ^ 1]) + original_header[1:])
assert checkpoint.verify([]).corrupt == [header]
header_path.unlink()
assert checkpoint.verify([]).missing == [header]
try:
    checkpoint.verify(["unknown"])
except tensorfs.NotFoundError:
    pass
else:
    raise AssertionError("an unknown component was accepted")
checkpoint.close()
try:
    checkpoint.verify()
except tensorfs.InvalidError:
    pass
else:
    raise AssertionError("a closed checkpoint was verified")
print("VerificationReport: missing, corrupt, component assets, header, read-only and no-fetch behavior passed")
