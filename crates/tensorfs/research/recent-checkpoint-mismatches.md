# Recent checkpoint interoperability failures

Researched 2026-10-19. The six-month window is 2026-04-19 through 2026-10-19.
Dates below are upstream merge dates, checked against primary GitHub metadata
and the corresponding source changes. These are documented upstream findings;
we did not run these models or reproduce their reported output quality.

## Confirmed fixes

### FLUX: a correct row permutation applied to the wrong LoRA factor

**2026-06-20 — [Diffusers #14001](https://github.com/huggingface/diffusers/pull/14001).**
The final adaptive-normalization projection needed its scale/shift output rows
reordered. For an update `B @ A`, that permutation belongs on B. The converter
applied it to both factors, changing the update despite valid tensor shapes.
The merged correction applies the transformation only to B.

[Immutable correction](https://github.com/huggingface/diffusers/blob/2d0110f8182d18834d5039b19232e5761023b5f6/src/diffusers/loaders/lora_conversion_utils.py#L544-L559).
This is a model/adapter-layout error, not a quantization error. The upstream PR
identifies silent corruption; it does not supply a reproduced image comparison
from our environment.

### MiniMax-H3: valid adapter matrices loaded at one-quarter strength

**2026-08-10 — [Diffusers #14409](https://github.com/huggingface/diffusers/pull/14409).**
A released H3 adapter mixes rank-16 and rank-64 matrices and omits explicit alpha
keys. Its intended scaling is one for every module. The loader chose a global
alpha from the first rank encountered; when that was 16, rank-64 updates received
a factor of `16/64 = 0.25`. Key ordering affected model behavior.

[Immutable correction and per-module defaults](https://github.com/huggingface/diffusers/blob/90c0ffdc045902a3667d473d2fbfc03e8716dba9/src/diffusers/utils/peft_utils.py#L156-L174).
Declared alphas retain precedence. This is adapter scaling metadata, not the
quantization of its stored values.

### NVFP4: scales reconstructed from already-packed bytes

**2026-08-03 — [NVIDIA Model Optimizer #2038](https://github.com/NVIDIA/Model-Optimizer/pull/2038).**
The exporter treated already-compressed NVFP4 storage as logical weights when
deriving scales. It produced half-length scale arrays with invalid values.
The correction preserves compression-time scales, reconciles their global
normalization, and handles a possible swizzled scale representation. Another
defect leaked an internal scale key that downstream loaders rejected.

[Immutable implementation](https://github.com/NVIDIA/Model-Optimizer/blob/b75227ce1bd143871c69d31330be5a327cfbdd7a/modelopt/torch/export/unified_export_hf.py#L544).
Upstream reports both load failures and incorrect output; these are distinct
outcomes. Explicit encoding metadata can expose missing roles or invalid
geometry, but cannot reconstruct an exporter’s lost original scales.

### MiniMax-M2: a config field changed the meaning of valid weights

**2026-09-17 — [Transformers #48486](https://github.com/huggingface/transformers/pull/48486).**
The original model config specified `rotary_dim=64` for a 128-dimensional head.
The receiving implementation did not translate that legacy field, applying
rotary encoding to the full head instead of half. The correction maps it into
the receiving implementation’s partial-rotation setting.

[Immutable config correction](https://github.com/huggingface/transformers/blob/da9d2c460c24cd9b1ee215d6e824601b48fbf9d4/src/transformers/models/minimax_m2/configuration_minimax_m2.py#L100-L105).
This is model/config semantics; changing a tensor decoder would not repair it.

### RT-DETR and SEW-D: loading returned a model with random parameters

**2026-09-14 — [Transformers #48744](https://github.com/huggingface/transformers/pull/48744).**
Incorrect base-model prefixes prevented checkpoint keys from matching the
target module tree. Upstream reports all 502 RT-DETR parameters and all 222
SEW-D parameters remaining randomly initialized for the affected loading paths.
The call returned with missing/unexpected-key warnings rather than failing.

[Immutable RT-DETR prefix correction](https://github.com/huggingface/transformers/blob/56c5e8768a956e395e3ccc3346ee91fafbc380d8/src/transformers/models/rt_detr/modeling_rt_detr.py#L993-L996).
This demonstrates an unsuccessful weight assignment, not corrupted stored bytes
or warning-free loading.

## Recent reports that remain unresolved

[ComfyUI #15114](https://github.com/Comfy-Org/ComfyUI/issues/15114), opened
2026-07-28, reports a direct-matrix LoKr path ignoring the adapter’s alpha/rank
scale: 316 modules matched, but strength one produced nearly uniform images.
The proposed correction remains open. Treat the output claim as the reporter’s
observation, not a merged and independently reproduced fix.

Other recent ComfyUI reports concern fused versus split Q/K/V adapter keys and
missing NVFP4 metadata. Their proposed fixes also remain open; they should not
be presented as confirmed merged fixes.

The older [WAN example #13238](https://github.com/Comfy-Org/ComfyUI/issues/13238)
was opened 2026-03-31 and closed 2026-04-01, outside this six-month window. The
SD3 example remains valid source evidence but is not a new incident.

## What this means for cozytensors

There are separate contracts for tensor encoding, model parameter layout,
adapter scaling, and model configuration. A self-describing tensor encoding
settles packing, scale roles and decoding formulas. It does not by itself
settle Q/K/V roles, LoRA-factor transformations or positional-encoding behavior.

The current cozytensors model class/configuration fields describe a model by
reference to an implementation. A stronger model-layout contract would need
framework-neutral, versioned semantics for parameter roles, axes, fused groups
and architecture-defining configuration. That is a design proposal, not an
implemented format feature or a claim that TensorFS supports every model above.

Even an explicit schema cannot establish that an exporter wrote the correct
learned values. Importer conformance and refusal of unsupported or ambiguous
conventions remain necessary. None of these examples demonstrates that a file
extension alone repairs an incorrect conversion.
