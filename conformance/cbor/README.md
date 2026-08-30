# CozyTensors CBOR conformance

This directory is the independent tfs-027 format gate. TensorFS Rust remains the only
production codec. Python (`cbor2==6.1.4`) and Go (`fxamacker/cbor/v2==v2.9.3`) independently
validate the closed schema, encode it with RFC 8949 Core Deterministic CBOR, strictly decode,
and require deterministic re-encoding to reproduce the resident bytes.

Neither implementation imports Runtime, Tensorhub, or TensorFS internals. They are format
conformance tools, not product parsers.

The diagnostic JSON accepted by all three seats is projection-only:

```text
{
  "format": "cozytensors/1",
  "configs": [[name, "sha256:<hex>", length, media_type], ...],
  "assets": [[name, "sha256:<logical>", length, media_type,
              [["sha256:<segment>", length], ...]], ...],
  "encodings": [{logical_dtypes, logical_rank?, roles, doc?, vectors?}, ...],
  "components": [[name, [[key, dtype_name, shape, encoding_index,
                          [[role, dtype_name, shape,
                            {"inline_hex": "..."} |
                            {"segments": [["sha256:<hex>", length], ...]}], ...]], ...]], ...]
}
```

Run the two independent seats:

```bash
uv run --project conformance/cbor/python \
  python scripts/cbor-conformance.py conform
```

Add the production Rust seat once `tfs cbor` is built:

```bash
uv run --project conformance/cbor/python \
  python scripts/cbor-conformance.py --tfs target/release/tfs conform
```

`conform --update` rewrites frozen binary vectors only after every available encoder agrees.
Ordinary conformance never rewrites them. Green vectors must encode and project identically;
red vectors must be refused by every seat.

## Real-header scaffold

The retired JSON header did not carry construction order, so conversion never guesses it.
Supply an explicit order document:

```json
{
  "components": [
    ["component_name", [["tensor.key", ["role_a", "role_b"]]]]
  ]
}
```

Then convert and measure codec process/size scaffolding:

```bash
uv run --project conformance/cbor/python \
  python scripts/cbor-conformance.py convert-legacy OLD_HEADER.json ORDER.json \
  --out h3-diagnostic.json

uv run --project conformance/cbor/python \
  python scripts/cbor-conformance.py --tfs target/release/tfs benchmark \
  h3-diagnostic.json --reps 7
```

The current local H3 source for that command is
`/home/fidika/cozy_v2/.artifacts/h3-bf16-adaln-pruned-evidence/header.json`
(3,547 tensors). Its order input must come from tfs-026's exact construction traversal;
this tool deliberately has no lexical, natural-sort, or carrier-order fallback.

The scaffold reports process-start-inclusive times. It is not the tfs-027 production Rust or
full-fill benchmark and must never be cited as such.
