# Pre-cut CBOR regression suite

This harness tests the pre-cut codec and its frozen vectors; it does not validate the redesigned
format. The authoritative [cozytensors specification](../../spec/cozytensors.md) defines that format.
The cutover replaces this suite with the new-format conformance corpus.

Run from the TensorFS directory. The independent Python codec imports no TensorFS internals;
add `--tfs` to compare the pre-cut Rust codec as well:

```bash
uv run --python 3.12 --project conformance/cbor/python \
  python scripts/cbor-conformance.py conform
uv run --python 3.12 --project conformance/cbor/python \
  python scripts/cbor-conformance.py --tfs target/release/tfs conform
```

Green vectors must reproduce the same bytes; red vectors must be refused. Ordinary runs never
rewrite vectors. `conform --update` rewrites them only after every available encoder agrees.
