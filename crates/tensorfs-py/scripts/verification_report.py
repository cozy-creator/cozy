"""The public Python report against an actual local checkpoint; no network or repair."""

import json
import hashlib
import shutil
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

# A known tree is checked before its repo-file references are trusted. Other variant
# weights are outside the selected checkpoint; explicit components ignore the tree.
repo_source = folder.parent / "repo-source"
shutil.copytree(folder, repo_source)
readme_bytes = b"repository notes"
(repo_source / "README.md").write_bytes(readme_bytes)
repo_cache = folder.parent / "repo-cache"
tree_id = tensorfs.upload_folder(repo_id="local/verification", folder_path=repo_source, cache_dir=repo_cache)
named = tensorfs.open(tree_id, local_files_only=True, cache_dir=repo_cache)

def blob(reference):
    digest = reference[0]
    return repo_cache / "blobs" / digest[:2] / digest[2:4] / digest

readme = (hashlib.sha256(readme_bytes).hexdigest(), len(readme_bytes))
blob(readme).unlink()
assert named.verify().missing == [readme]
assert named.verify(["model"]).ok
tree = named.tree_ref
original_tree = blob(tree).read_bytes()
altered_tree = original_tree.replace(readme[0].encode(), b"0" * 64)
assert len(altered_tree) == len(original_tree) and json.loads(altered_tree)
blob(tree).write_bytes(altered_tree)
blob(model).unlink()
report = named.verify()
assert report.corrupt == [tree] and report.missing == [model]
assert named.verify(["other"]).ok
assert blob(tree).read_bytes() == altered_tree
blob(tree).unlink()
report = named.verify()
assert set(report.missing) == {tree, model} and report.corrupt == []
named.close()

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
