# tensorfs (v2)

Canonical tensor byte plane: CAS, cozytensors headers, snapshots, ingest. Design authority:
[tracker-v2](https://github.com/cozy-creator/tracker-v2) `tensorfs.md` / `cozytensors.md`; issues `tracker/tensorfs/`.
Rust workspace: `tensorfs-core` (torch/CUDA/HTTP/SQL-free), `tensorfs-layout`, `tensorfs-py` (marshal-only PyO3, later).
No automated tests (decisions.md #160): verification = live runs + benchmarks.

## tfs (tfs-013)

```
cargo run --release -p tensorfs-core --bin tfs -- registry        # seed specs: alias -> digest
cargo run --release -p tensorfs-core --bin tfs -- vectors verify  # run the frozen corpus live
cargo run --release -p tensorfs-core --bin tfs -- vectors gen     # rewrite it (ids change: review the diff)
cargo run --release -p tensorfs-core --bin tfs -- bench 3699 50   # writer throughput + peak RSS
cargo run --release -p tensorfs-core --bin tfs -- flip            # identity-flip matrix
cargo run --release -p tensorfs-core --bin tfs -- id <file>       # object id + derived ids
```

## checkpoints (tfs-015)

The real end-to-end plane: a synthetic multi-component tree is objectized through the CAS,
declared by one canonical `model.cozytensors`, and reconstructed from the header alone.

```
tfs checkpoint write <root>            # ~382 MiB, 3 components, 4 encodings -> header_digest
tfs checkpoint read <root> <hex>       # every anchor equation, every segment, byte-identical
tfs checkpoint info <root> <hex>       # component/scope query — opens ONE object, the header
tfs checkpoint locate <root> <hex> <component> <key> <role> <off> <len>
tfs checkpoint admit <root> <hex>      # serving admission against the local capability pin
tfs checkpoint caps | size [n]         # whole-document total caps; uniform-form header cost
scripts/checkpoint-runs.sh all         # the whole thing plus five planted red arms
```

## snapshots (tfs-004)

```
tfs snapshot show <root> <hex>      # strict decode + decode->re-encode byte identity
tfs snapshot walk <root> <hex>      # transitive root; every-and-only vs the store census
tfs snapshot verify <root> <hex>    # full-checkpoint verify: every declared byte
tfs snapshot flip <root> <hex>      # what moves checkpoint_id / header_digest / topology
tfs snapshot fuzz <root> <hex>      # bounded mutation fuzz over canonical bytes
tfs snapshot blob put|read|repair <root> <file> [i]
scripts/snapshot-runs.sh all        # the whole plane plus its red arms
```

`checkpoint_id = snapshot_id`, unconditionally. Contained documents (EncodingSpec objects
and their vector sets) live at `closure/<digest>.json`, where the path terminal IS the
digest; the header attachment is TYPED and must be projected at exactly one path.

## ingest (tfs-003)

The only normal tensor-ingest door. `inspect` and `plan` read HEADERS only — every
structural refusal is decidable for a few hundred KiB, before a GPU is rented. `run` hands
the job to an isolated child; `install` is the coordinator's separate act; everything that
never reaches it reaps.

```
tfs ingest inspect --source unet=<file> [--source vae=<file> ...]   # hermetic bounded decode
tfs ingest bank <dialect> <family> <variant> <converter> --source c=<file>   # the REVIEW act
tfs ingest plan  <component=alias,...> --source c=<file>            # zero tensor bytes read
tfs ingest run   <store> <component=alias,...> --source c=<file> [--config c=<file>]
tfs ingest install <store> <session>        # the coordinator promotes ONE root
tfs ingest roots|reap <store> [--now]       # candidates are not checkpoints
tfs ingest reingest <store> <header> <target> [--rekey c=from:to]   # inherit by reference
tfs ingest converters | golden              # reviewed converters + their geometry, live
scripts/ingest-runs.sh all                  # the border plus 26 planted red arms
```

A `--source` names its component; a FILENAME never authorizes anything. Classification is
a key-set fingerprint against a reviewed registry (`vectors/fingerprints/`), and both
absence and ambiguity refuse. Measured on a real 6.94 GB SDXL fp16 tree: **84 MB/s
end-to-end at a 13.2 MiB peak RSS** (streaming, never whole-file), planned in 30 ms from
336 KiB of headers. Re-ingesting an already-canonical component inherits every object by
reference: 167 MB carried, **zero bytes hashed**.

## registry (tfs-008)

```
tfs structure list | match <root> <hex>       # banked fixtures, and what a checkpoint is
tfs structure bank <rows> <family> <variant>  # bank transcribed evidence
tfs conform [alias]                           # qualify implementations against vectors
tfs capability <digest> <device> [--alias a]  # fail-closed admission
scripts/registry-runs.sh all                  # fixtures, mining, conformance, red arms
```

Vectors are PRODUCER-MINED: `scripts/mine-vectors.py` runs v1's own quantizers (the quarry
under `~/cozy`) and transcribes raw bytes; `tfs vectors mine` canonicalises them. A
vectorless spec stores and inspects and can never qualify an implementation.

`vectors/` is frozen DATA: `closure/` holds the platform EncodingSpec objects, `fingerprints/`
the reviewed classification registry, `cases/<group>/`
holds each vector's exact stored bytes plus a `.verdict` (expected outcome, frozen ids),
`mined/` holds the producer-mined conformance vectors and `structure/` the banked
structure fixtures. `vectors gen` regenerates only `cases/` and `closure/` — the other two
are evidence with their own provenance and their own banking commands.
Caps in `limits.rs` are PROVISIONAL until proto-001's real H3+SDXL corpus ratifies them.
