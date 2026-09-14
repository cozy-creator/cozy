# Real H3 style adapter headers

These are only the safetensors length prefix and original JSON header, range-read
from the immutable revisions in MANIFEST.json. They contain no tensor payloads.
`source_sha256` is Hugging Face's declared whole-file LFS digest; `header_sha256`
is locally measured over the retained bytes. Neither is full-body verification.

The spatial variants contain 416 FP16 factors: rank16 A/B pairs for four native
projections in 50 main blocks and two token-refiner blocks. AdaLN factors were
removed upstream. Wushu variants contain 600 FP32 tensors: rank32 down/up pairs
plus scalar alpha for four native projections in 50 main blocks. Its metadata
reports FL2VA training. The filename's `adaln_full` text does not describe tensors
present in this file.

Every target shape fits both native H3 DiTs, but structural fit does not establish
REF2VA behavior or quality transfer. The same adapter is never implicitly applied
to both components. The canonical conversion splits head-interleaved QKV B rows,
swaps native SwiGLU B halves, preserves A and alpha, and changes no scalar values.
It uses the reviewed H3 geometry (56 heads, head dimension128); other geometry
refuses. Runtime validates the chosen base component and applies request scales.

The two reviewed keysets classify independent of filenames; multiple matching
versions refuse and list candidate members. Exact named source profiles for
other families retain their existing member constraints.

Reproduce the registry (header-only, no downloads):

```sh
cargo build -p tensorfs-core --bin tfs -j 2
python3 scripts/rebank-h3-lora.py --tfs target/debug/tfs
cargo test -p tensorfs-core --test h3_lora -j 2
```

The byte-level test additionally drives generated row-tagged carriers through the
real native conversion writer and replay, checking all output values against an
independent Q/K/V and SwiGLU reference. This is converter validation, not inference
quality evidence.
