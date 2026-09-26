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

Creator now parses and canonicalizes this sidecar without putting `lora` into a
package request schema. The parsed `RunKeys.Overlays` is deliberately kept
separate from the legacy exact `ModelRef.Adapters` rows until package metadata
can prove the target component and weight range. A model overlay must never guess
a component or use a manifest digest as a compatibility claim.

The remaining admission seam is explicit. PackageInterface slots need an
opt-in compatibility descriptor (adapter kind, base family, target components,
capability range and per-component weight bounds), and the Hub model card needs
the adapter's declared family/target/capability metadata. Current cards expose
only repository family and lane components, which is insufficient to validate a
generic adapter safely. Until that metadata is present, a generic overlay must
refuse before transfer; explicit `turbo_lora` slots remain independent bindings.

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
