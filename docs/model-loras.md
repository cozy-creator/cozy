# LoRAs on model slots

A request can add an ordered stack of compatible LoRA checkpoints to a model
parameter without changing package code. Each selection names an explicit model
component and a strength:

```sh
cozy run paul/minimax-h3/fl2va prompt='A martial artist practices in a courtyard' \
  --lora 'model:fl2va_dit=paul/minimax-h3-spatial-physics-lora#sha256:7c5e4d22443d9537ba5befd79f133604e9caf68ed06ddcf645af4476bf60a742,0.3' \
  --lora 'model:fl2va_dit=paul/minimax-h3-wushu-action-v7-lora#sha256:f8c8150d3ab44314b1d54085ec1bdf23c2d4b198edc160a98222bcde1d863cf7,0.5'
```

Use the parameter name shown by the callable's description. For H3 REF2VA, the
explicit target is `ref2va_dit`. Structural compatibility is checked against the
actual selected base component; training on FL2VA alone does not establish REF2VA
quality. The order of repeated flags is retained. Strength defaults to1 when the
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
ordinary adapter downloads require the complete wire55 Host/Runtime/Creator cohort;
older workers refuse before preparation. Job-only model inputs do not execute and
therefore do not accept serving adapters.

Runtime owns unmerged application and request cleanup. Base weights are never
folded or rewritten per strength. The ordinary base, adapter checkpoints and scales
all remain explicit reproduction facts.
