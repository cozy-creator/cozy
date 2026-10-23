# Source-header fixtures

These fixtures contain safetensors length prefixes and JSON headers, plus shard indexes where
needed. They contain no trained tensor payloads. Each directory's `MANIFEST.json` pins source
revisions, file identities, lengths and measured header hashes. Provider-declared whole-file
hashes are not locally verified payload hashes; sparse carriers prove geometry, not values or
inference quality. Conversion checks must use an independent source-to-output oracle.

| Directory | Coverage |
|---|---|
| [anima-civitai-headers](anima-civitai-headers/) | Anima single-file transformer, text encoder and Qwen-Image VAE; mapping follows the upstream Diffusers converter. |
| [anima-hf-headers](anima-hf-headers/) | The corresponding Diffusers folder and its source key sets. |
| [civitai-sdxl-headers](civitai-sdxl-headers/) | A Civitai SDXL single file; its source body was independently downloaded and hash-verified, but only its header is retained here. |
| [sdxl-headers](sdxl-headers/) | The upstream SDXL folder, including per-member identities and shapes. |

H3 normalization has two distinct source contracts: native base QKV rows are interleaved per
head; the reviewed trainer adapter B factors contain contiguous Q/K/V thirds. Native SwiGLU
`[gate; value]` halves become Diffusers `[value; gate]`, including adapter B halves; A and alpha
are unchanged. Names and shapes cannot prove which source layout a producer used.

Preserve the source terms with their fixtures: Anima's
[license](anima-civitai-headers/LICENSE.md) is also retained with its
[Diffusers excerpts](anima-hf-headers/LICENSE.md).
