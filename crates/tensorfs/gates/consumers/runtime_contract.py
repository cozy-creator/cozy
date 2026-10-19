#!/usr/bin/env python3
"""Hosted installed-Runtime metadata/snapshot/wire compatibility; no Torch or GPU work."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time

import msgspec

from cozy_runtime.author import App, Invocation, Outputs, Tree, attempt, describe, script_app
from cozy_runtime.internal import canonical
from cozy_runtime.internal.executor import _serialize_result

from test_tooling import Contract
from validate import tree_document


class Request(msgspec.Struct):
    pass


class Result(msgspec.Struct):
    files: Tree


def main():
    assert importlib.util.find_spec('torch') is None, 'this metadata check needs no Torch environment'
    for native, hostpin in [(False, False), (True, False), (True, True)]:
        test = Contract()
        test.setUp()
        try:
            test.fixture(native=native, hostpin=hostpin)
            sys.path.insert(0, str(test.root))
            app = script_app('payload')
            surfaces = describe(app)
            assert len(surfaces) == 1 and surfaces[0].name == 'main' and surfaces[0].accelerator
            fields = msgspec.structs.fields(surfaces[0].payload_type)
            assert [f.name for f in fields] == (['native_wheel', 'hostpin_bundle'] if hostpin else ['native_wheel'])
            assert all(f.type.__name__ == 'FileAsset' for f in fields)
        finally:
            sys.path.remove(str(test.root))
            test.doCleanups()
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        source = root / 'source'
        source.mkdir()
        (source / 'a.json').write_bytes(b'{"rows":2}')
        (source / 'nested').mkdir()
        (source / 'nested' / 'caf\u00e9.txt').write_bytes(b'12345678')
        original = tree_document(source)
        # Runtime's actual canonical serializer must agree with the traced TensorD tree
        # encoding: sorted keys/paths, UTF-8, compact, no newline; no float values here.
        assert canonical.write_application(json.loads(original)) == original
        app = App()

        @app.entrypoint(accelerator=False)
        def main(payload: Request, out: Outputs) -> Result:
            snapshot = out.save_tree(source)
            (source / 'a.json').write_text('changed after snapshot')
            return Result(snapshot)

        spool = root / 'spool'
        envelope, outcome, _ = attempt(app.get('main'), {}, Invocation('987', spool, time.monotonic() + 30))
        assert outcome.terminal == 'succeeded', outcome
        assert envelope is not None
        reference, outputs, oversize = _serialize_result(envelope, spool)
        assert reference and oversize == 0 and len(outputs) == 1
        row = outputs[0]
        assert row['output_id'] == 'files' and row['kind'] == 'tree'
        assert row['asset_ref'].rsplit('/', 2)[-2] == 'tree'
        assert str(root) not in json.dumps(outputs)
        snapshot = envelope.result.files.path
        assert tree_document(snapshot) == original
        assert canonical.write_application(json.loads(tree_document(snapshot))) == original
    print(json.dumps({'ok': True, 'proof': 'installed-runtime-consumer-tooling-contract',
                      'script_accelerator': True, 'input': 'FileAsset',
                      'payload_fileasset_counts': [1, 1, 2],
                      'snapshot_and_tree_result': True, 'canonical_tree_bytes': True,
                      'scope': 'CPU metadata/snapshot/serialization only; no GPU execution'}))


if __name__ == '__main__':
    main()
