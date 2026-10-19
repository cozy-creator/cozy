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
| [h3-lora-headers](h3-lora-headers/) | Kohya and A/B trainer adapters, attention-only variants and AdaLN refusal cases. |
| [h3-pdd-headers](h3-pdd-headers/) | Two 728-tensor BF16 adapters with rank-64 factors and 32-head banks; the producer owns head collapse and the serving overlay. |
| [instrument-headers](instrument-headers/) | Evaluation Qwen/Whisper sources and shard geometry; constructor order comes from Eval's model factories, with no factory execution in TensorFS. |
| [qwen-image21-headers](qwen-image21-headers/) | Qwen-Image-2.1 component headers, unchanged shard indexes and construction order. |

H3 normalization has two distinct source contracts: native base QKV rows are interleaved per
head; the reviewed trainer adapter B factors contain contiguous Q/K/V thirds. Native SwiGLU
`[gate; value]` halves become Diffusers `[value; gate]`, including adapter B halves; A and alpha
are unchanged. Names and shapes cannot prove which source layout a producer used. The fixture
manifest records the selected contract or expected refusal, and structural fit alone does not
establish adapter quality on another component. Eval owns its instruments' tied destinations,
rotary buffers, casts and processor configuration.

Preserve the source terms with their fixtures: Anima's
[license](anima-civitai-headers/LICENSE.md) is also retained with its
[Diffusers excerpts](anima-hf-headers/LICENSE.md); Qwen-Image has its
[license](qwen-image21-headers/LICENSE) and [Notice](qwen-image21-headers/Notice).
