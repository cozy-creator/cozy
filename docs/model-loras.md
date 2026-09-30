# LoRAs on model slots

A request can add an ordered stack of compatible LoRA checkpoints to a model
parameter without changing package code. Each selection names an explicit model
component and a strength:

```sh
cozy run paul/minimax-h3/fl2va prompt='A martial artist practices in a courtyard' \
  --lora 'model:fl2va_dit=paul/minimax-h3-spatial-physics-lora,0.3' \
  --lora 'model:fl2va_dit=paul/minimax-h3-wushu-action-v7-lora,0.5'
```

LoRA references use the ordinary checkpoint resolver. Omit the release to select
the latest published release, or append `@1.0.0` to choose one. A sole lane resolves
automatically; users do not need hashes or a lane matching the base weight dtype.
For example, FP16 adapter factors can be applied to an FP8 base if their tensor
targets and dimensions match. Multiple distinct lanes remain an explicit ambiguity
until compatibility selection is available; the resolver does not guess by name.

Use the parameter name shown by the callable's description. For H3 REF2VA, the
explicit target is `ref2va_dit`. Structural compatibility is checked against the
actual selected base component; training on FL2VA alone does not establish REF2VA
quality. The order of repeated flags is retained. Strength defaults to 1 when the
comma suffix is omitted; zero and negative finite strengths are valid and are not
normalized across the stack. Adapter alpha/rank scaling is additional to that
request strength.

A composition's child serving function can be selected by its full model slot:

```sh
cozy run paul/minimax-h3/long_form --input scene.yaml \
  --lora 'motion_segment_turbo.models.base_model:ref2va_dit=paul/style@1.0.0,0.5'
```

`<function>.models.<parameter>` names that captured function in the root package.
Use `<package>/<function>.models.<parameter>` for an external captured callable.
These selections apply to that function's calls and never to another function
that happens to use the same parameter name. The same selectors work for ordinary
`model.<slot>=<checkpoint>` overrides. A bare parameter still names the root.

JSON/YAML may use `models.<slot>.lora` as an ordered list of `ref`, `weight`, and
optional `component` entries. Runtime infers an omitted component only when one
compatible component exists. An adapter-only selection keeps the ordinary base
model default. Provider references use the same machine-side source preparation
as checkpoints; list optional reviewed `profiles` and `source_component` in each
overlay when needed. The authored function receives none of these controls.
When combined, JSON/YAML and inline overlay lists come first; repeated `--lora`
flags append in their command-line order, including equivalent slot spellings.

For a foreign adapter, select its tensor file explicitly when the repository has
multiple versions:

```sh
cozy model download 'https://huggingface.co/OWNER/REPO/blob/main/FILE.safetensors' local/style
cozy model upload local/style your-account/style
```

The upload returns a digest-pinned checkpoint reference usable without publishing
a release label. A repository-only URL with several matching checkpoint files
refuses and lists the candidates. Ingest recognizes reviewed tensor structures,
not EXE files, workflow JSON or repository names.

The API representation is an `adapters` list inside each selected model row. An
input may name an open catalog selection or immutable provider source; Runtime
resolves exact checkpoint identities before execution. Each stack entry retains
its component, source component, and canonical finite decimal scale. Both ordinary
and native retained bases keep their original custody. The executing Runtime must
advertise `model_overrides`, and the Host must support wire70 so it forwards both
release-root and captured choices. Unsupported peers refuse this operation before submission;
base-only requests remain supported. Job roots may select their captured serving
children without declaring a Model argument of their own.

Runtime owns unmerged application and request cleanup. Base weights are never
folded or rewritten per strength. The ordinary base, adapter checkpoints and scales
all remain explicit reproduction facts.
