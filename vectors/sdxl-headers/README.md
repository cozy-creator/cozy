# Reviewed SDXL source headers

These are bounded header reads from
`stabilityai/stable-diffusion-xl-base-1.0@462165984030d82259a11f4367a4eed129e94a7b`.
`MANIFEST.json` records each exact member, provider-declared full-file digest and
length, and independently hashed header prefix. No weight payload is included.

The native `ingest bank` result matches every field of the original reviewed
SDXL observations, including their header-and-geometry `source_sha256`, except
the schema digest left stale by the schema projection change. Key sets, dtypes,
shapes, converters, and source-profile construction order are unchanged.

`scripts/rebank-sdxl.py --tfs target/debug/tfs` proves that relationship. Its
sparse temporary carriers represent declared geometry for header inspection;
their unfilled bodies are never used as weight or model-quality evidence.
