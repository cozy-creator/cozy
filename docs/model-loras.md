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

The API representation is an `adapters` list inside each selected model row. Every
row carries `component`, immutable `model`/`manifest` (and optional release/lane),
`source_component`, and a canonical finite decimal `scale` string. Both ordinary
and native retained bases keep their original custody. Native binding support and
ordinary adapter downloads require the complete wire57 Host/Runtime/Creator cohort;
older workers refuse before preparation. Job-only model inputs do not execute and
therefore do not accept serving adapters.

Runtime owns unmerged application and request cleanup. Base weights are never
folded or rewritten per strength. The ordinary base, adapter checkpoints and scales
all remain explicit reproduction facts.
