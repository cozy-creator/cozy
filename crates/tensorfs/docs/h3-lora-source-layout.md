# H3 trainer adapter QKV layout

H3's original base checkpoint and its trainer's saved adapter use different fused QKV row layouts. TensorFS must distinguish them when splitting a fused projection into Diffusers `to_q`, `to_k`, and `to_v`.

## Source contract

The [official Diffusers converter](https://github.com/huggingface/diffusers/blob/c60830ee365d520ab52b110dda562dd26f7b4d7f/scripts/convert_minimax_h3_to_diffusers.py#L110) distinguishes two layouts:

* Original MiniMax checkpoint: each head's Q, K, and V rows are adjacent. Its loader reorders these into three contiguous projection matrices.
* Loaded reference model and its state dict: all Q rows, followed by all K rows, followed by all V rows.

The [trainer snapshot from September 3](https://github.com/AkaneTendo25/musubi-tuner/blob/b40ed86ebc028fa30b3302c379a104f033f2a1af/src/musubi_tuner/minimax_h3/model.py#L298), before the selected Wushu adapter's September 5 training, computes the fused projection, chunks its output into three contiguous thirds, and then unflattens each into heads. Its [adapter export](https://github.com/AkaneTendo25/musubi-tuner/blob/b40ed86ebc028fa30b3302c379a104f033f2a1af/src/musubi_tuner/networks/lora.py#L948) does not reorder QKV factors. Wushu's embedded metadata names `networks.lora_minimax_h3`.

The [Comfy H3 attention](https://github.com/Comfy-Org/ComfyUI/blob/a25c7bf2b8c7408d8724f4245dbe09d95992e3a1/comfy/ldm/minimax/model.py#L169) also consumes contiguous Q/K/V thirds. Its generic LoRA key mapping attaches the fused factor directly to that projection. Diffusers consumes separate Q/K/V projection outputs and unflattens each into contiguous heads.

The `h3.native/2` base checkpoint converter retains its existing head-interleaved seam. Trainer LoRA conversion must split B into contiguous thirds. A is shared unchanged. Alpha is unchanged. The SwiGLU B halves still swap from source gate/value order to Diffusers value/gate order.

## Exact artifact audit

The independent audit reads complete immutable upstream safetensors and every canonical CAS part. It verifies full source hashes, CAS hashes, shapes, dtypes, finiteness, and every factor byte without calling the TensorFS converter or using its expected output fixtures.

| Adapter | Pinned upstream commit | Source SHA256 | Canonical tensors | Incorrect QKV B tensors |
| --- | --- | --- | ---: | ---: |
| Wushu v7 2000step | `9598ef0b02f4590e202e956b4e6328dc71ae2bf5` | `2fdc4a990591abfeb6fa4bc83b7c1974dcac0a6132c982ef386cd6243790be64` | 900 | 150 |
| Spatial Physics clean3000 | `476b24df28b9b7cc5481b750469698f5cc4b0558` | `7d14f3701560068e7004159c8b2a7278bd2dbfc9e5e3b60d0bc9aef6c049919d` | 624 | 156 |

The published canonical checkpoints are respectively `f8c8150d3ab44314b1d54085ec1bdf23c2d4b198edc160a98222bcde1d863cf7` and `7c5e4d22443d9537ba5befd79f133604e9caf68ed06ddcf645af4476bf60a742`.

All canonical bytes match the existing `h3.lora/1` transform. Hashes show no payload corruption. That transform misroutes 33,996,800 of 34,406,400 Wushu QKV B values and 17,676,423 of 17,891,328 Physics QKV B values relative to the upstream projection contract. All other factors match the intended conversion.

For example, destination Q head 1 row 0 must use raw B row 128. The existing converter uses row 384. Destination K head 0 row 0 must use raw B row 7168; the existing converter uses row 128. Every shape remains valid, so structural tests and correct Runtime PEFT arithmetic cannot detect this error.

## Repair and publication

The correction needs an independent projected-Q/K/V regression that computes the upstream fused projection before dividing its output into thirds. A fixture that repeats the mapper's row formula is insufficient.

Use a new converter and rebanked profile identity for corrected conversions. Preserve existing checkpoint objects and published releases. Corrected public adapters require new immutable patch releases and a fresh GPU quality comparison; the previous videos exercised the wrong normalized factors and cannot establish adapter quality.

## Declare the source layout before normalization

The reviewed `h3/native-lora-a-b/2` and `h3/native-lora-kohya/2` profiles have
`auto_select: false`. Automatic planning and header preflight refuse a matching
fused adapter until the caller selects its reviewed format explicitly. Contiguous
and head-interleaved QKV factors can have identical names, dtypes and shapes, so
the header fingerprint cannot choose between them. An explicit raw `as-is/1`
selection still preserves bytes without interpreting their layout.

Both profiles match by grammar, not by an exact key list: every key carries the
profile's prefix (`diffusion_model.` or `lora_unet_`) and the `h3.lora/2` planner
accepts it. That admits any non-empty set of complete A/B pairs with an optional
scalar alpha on the reviewed projections, so attention-only fal trainer exports
convert. Unknown keys, half pairs and A/B rank disagreement refuse; a carrier the
grammar rejects is stored as-is when no profile is declared. AdaLN factors
(`adaln_proj`) refuse `ADAPTER_TARGET_UNSUPPORTED`: the converter has no mapping
for them and dropping them would change the adapter. The refusal counts them and
names the first four.

Not verified: for fal-trainer exports the fused QKV rows are assumed to be
contiguous thirds, the layout `/2` declares. That rests on the vh5tape ComfyUI
build copying the factors bit-exact onto Comfy's contiguous `qkv_proj`. Unlike the
Wushu and Physics adapters, these have no projected-output or GPU check.

Selecting `/2` declares the trainer/Comfy contiguous-QKV contract. Before choosing
it for an unknown producer, review the exporter's declaration and loading code
and check equivalent projected Q/K/V outputs independently. Explicit selection
does not prove arbitrary misdeclared bytes satisfy that contract. The pinned
Wushu and Physics files above have that source and numerical qualification.

CozyTensors' numeric `EncodingSpec` describes each canonical tensor's physical
representation. The canonical separate `to_q`, `to_k`, and `to_v` keys declare
the projection identities; normalization must establish the source-to-canonical
mapping before those tensors exist. This correction adds no numeric encoding or
deployment-equality requirement. Other source profiles retain automatic
selection by default.
