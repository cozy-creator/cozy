# Reviewed Civitai SDXL header

This is the exact8-byte Safetensors prefix and JSON header from Civitai version128078,
file92696, already banked as `civitai-model-101055-version-128078`. It was read from
the fully downloaded, SHA256-verified native source on an owned private worker.
The source body is6,938,078,334bytes; no trained tensor payload is included here.
MANIFEST.json records both full-source identity and bounded header identity.

The current native schema projection changed the four previously banked schema
hashes; every other reviewed fingerprint field is identical. CI reconstructs a
sparse, header-only carrier and checks the exact bank/profile without reading
its absent model body. Altered dtype or missing key still refuses before conversion.
