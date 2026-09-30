# Generic weighted model-slot overlays

This note records the first bounded delivery of the generic overlay surface. It
is a model-binding sidecar, not a callable payload field:

```json
{
  "prompt": "...",
  "models": {
    "base_model": {
      "ref": "org/base@release",
      "lora": [
        {"ref": "org/style-a@release", "weight": 0.5},
        {"ref": "org/style-b@release", "weight": -0.25}
      ]
    }
  }
}
```

The list order is semantic and is retained in the request identity. Weights are
canonical finite decimal strings before submission, so `0.50` and `0.5` have one
identity. Inline callers can use `model.<slot>.lora:=<json-list>` or repeat the
short form `model.<slot>.lora=<ref>,weight=<number>`; the bare
`<slot>.lora=...` spelling is accepted because dotted payload fields are not
declared. The existing `--lora model-parameter:component=ref[,strength]` path
remains the component-explicit compatibility form.

Creator carries this sidecar as ordered `ModelRef.Adapters` and wire70
`ModelChoice.adapters`, outside the package request schema. Runtime owns source
resolution and compatibility against the actual base component and adapter factors.
An omitted component is inferred only when one compatible target exists; ambiguity
requires an explicit `component`. A manifest digest establishes bytes, not compatibility.

The `model_overrides` workspace capability gates adapter-bearing requests and exact
captured-callable selectors. Published roots carry the choices in `ReleaseRoot.models`;
unpublished captures retain them in `MachineExecutionCapture.model_choices` and its
digest. Private serving preparation resolves the same choices before returning its
fixed placement. The native/download base custody remains unchanged. Explicit
`turbo_lora` model parameters remain independent bindings.

Preparation must apply an accepted ordered overlay once to an isolated model
construction before compile/graph capture. The base construction must be restored
or discarded after the request; a no-overlay request must see pristine weights.
Reusable prepared variants are keyed by the base binding plus ordered adapter
references, weights and compatibility metadata. Concurrent requests cannot mutate
one shared base construction.

Composition is a capability, not an unconditional promise. A quantized base such
as H3 FP8 cannot safely be merged by dequantizing and requantizing arbitrary
weights; that would change the stored representation and needs its own numerical
qualification. A model class may instead provide a supported composer that clones
or fuses effective layer parameters into an isolated prepared graph while leaving
the pristine base bytes untouched. If no such composer is declared, admission
refuses that overlay for that model slot.

MiniMax H3's current `turbo_lora` path is a specialized exception: its hooks are
armed permanently and `LoRAFactors.accumulate` computes an update on each DiT
forward. That path is not generic bake-once qualification. A follow-up must either
materialize factors into an isolated prepared H3 construction (and precompute its
Turbo tables/heads) or keep the path explicitly separate from generic overlays.
