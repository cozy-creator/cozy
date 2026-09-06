#!/usr/bin/env python3
"""Consume the opt-in native source fixture in fresh Python processes.

Create a fresh owned fixture first:
  TFS_SOURCE_RESTART_FIXTURE=/absolute/fresh/path cargo test -p tensorfs-core --lib \
    ingest::source::tests::export_source_operation_restart_fixture -- --ignored --exact
Then run this script with the installed wheel's Python and that path.
The fixture is consumed: native lifecycle release and GC remove its retained source bytes.
"""
from __future__ import annotations

import json
import io
from pathlib import Path
import subprocess
import sys

import tensorfs


def observe(fixture: Path) -> dict[str, object]:
    metadata = json.loads((fixture / "restart-fixture.json").read_text())
    store = tensorfs.Store.open(metadata["store"])
    operations = store.model_source_operations()
    heads = []
    for head in metadata["heads"]:
        page = store.source_checkpoint_page(
            head["head_id"], head["head_length"], operation_id=metadata["operation"],
            slot=head["slot"], plan_digest=head["plan_digest"],
        )
        heads.append({"head": head["head_id"], "index": page["index"], "bytes": page["bytes"]})
    return {"operations": operations, "heads": heads}


def main() -> None:
    fixture = Path(sys.argv[1]).resolve()
    if len(sys.argv) > 2 and sys.argv[2] == "observe":
        print(json.dumps(observe(fixture), sort_keys=True))
        return
    metadata = json.loads((fixture / "restart-fixture.json").read_text())
    before = observe(fixture)
    assert before["operations"] == sorted((metadata["operation"], metadata["open_operation"]))
    store = tensorfs.Store.open(metadata["store"])
    plain = next(digest for alias, digest in tensorfs.seed_digests() if alias == "plain/1")
    transaction = "sha256:" + "84" * 32
    writer = store.begin_derived(transaction, 1, {}, {"new": {"drop": [], "add": {
        "weight": {"logical_dtype": "f32", "shape": [512], "encoding": plain,
                   "parts": {"value": {"dtype": "f32", "shape": [512]}}},
    }}}, {}, [("new", "weight")], 2048)
    assert store.model_source_operations() == before["operations"]
    writer.add_part("new", "weight", "value", io.BytesIO(bytes(2048)))
    writer.commit()
    store.derived_adopt(transaction, "unrelated-derived-root")
    replacement = json.loads(subprocess.check_output(
        [sys.executable, __file__, str(fixture), "observe"], text=True,
    ))
    assert replacement == before
    store.derived_dispose(transaction)
    for operation in store.model_source_operations():
        store.release_model_source(operation)
        store.release_model_source(operation)
        assert not (Path(store.root) / "tmp/ingest" / operation).exists()
    assert store.model_source_operations() == []
    result = tensorfs.gc(store.root)
    for head in metadata["heads"]:
        assert not store.contains(head["head_id"])
    print(json.dumps({"version": tensorfs.__version__, "reopened": replacement,
                      "released_roster": [], "gc": result}, sort_keys=True))


if __name__ == "__main__":
    main()
