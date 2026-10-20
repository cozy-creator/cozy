# Readable format examples

`header-basic.json` and `header-configs.json` are diagnostic JSON projections of CBOR headers.
A value such as `{"hex": "…"}` denotes a CBOR byte string, not an on-disk JSON object.
Real `.cozytensors` headers use the binary CBOR encoding in [the specification](../cozytensors.md).

`tree-basic.json` and `repo-basic.json` show the JSON document forms. Some chunk and asset
references are synthetic fixture identities; these examples are not a complete downloadable model.
The repo's public tag `1.1` and private tag `main` name the exact example tree bytes.

[Schema files](../schema/) describe structure; the specification also defines canonical encoding,
duplicate-key refusal, bounds, reference closure, construction order and decoding requirements.
