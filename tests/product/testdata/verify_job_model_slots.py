"""Check the real captured script selection against Runtime and native model bytes."""

import json
import sys
from pathlib import Path

from cozy_runtime.internal.worker.package_prepare import PreparationRefusal, _entrypoints


def refuse_construction(*args):
    raise AssertionError("job model preparation must not construct an inference model")


interface = json.loads(Path(sys.argv[1]).read_text())
captured = json.loads(Path(sys.argv[2]).read_text())
selections = [
    {
        "package": row["package"],
        "slot": row.get("binding_path") or row["slot"],
        "model": row["model"],
        "release": row.get("release", ""),
        "lane": row.get("lane", ""),
        "manifest": row["manifest"],
    }
    for row in captured
]
arguments = dict(
    package_name=captured[0]["package"],
    tensorfs_root=Path(sys.argv[3]),
    verified=lambda *_: None,
    derive=refuse_construction,
)
models, serving = _entrypoints(interface, selections=selections, **arguments)
assert len(models) == len(captured) and not serving
assert all(row["binding_path"] != row["slot"] for row in captured)

# Bare argument names are correct for invocation inputs, but not preparation.
bare = [dict(row, slot=source["slot"]) for row, source in zip(selections, captured)]
try:
    _entrypoints(interface, selections=bare, **arguments)
except PreparationRefusal as exc:
    assert exc.code == "package_prepare_model_selection_mismatch", exc
else:
    raise AssertionError("Runtime accepted bare argument names as model binding paths")
