# Cozy Video source format

`cozy.video/1` is an editable YAML authoring document. Creator composes it into ordinary H3
requests followed by one CPU `assemble_video` request; endpoints and the workflow engine never
parse YAML.

```yaml
format: cozy.video/1
shots:
  - id: establishing
    prompt: |
      Wide establishing shot of a night train entering a mountain station.
      Snow moves through warm platform light; realistic synchronized ambience.
    seed: 4101
    reference_media_to_video:
      references:
        - image: references/character.png
        - video: references/motion.mp4
        - audio: references/voice.wav

  - id: boarding
    prompt: |
      Continue from the exact prior frame as the traveler boards the train.
      Preserve clothing, face, lighting direction, camera motion, and station geography.
    seed: 4102
    first_last_frame_to_video:
      first_frame: previous

assembly:
  audio: segments
```

A source has 2–8 ordered shots, unique editor IDs, non-empty prompts of at most 4,096 Unicode
characters, and explicit decimal int64 seeds. Each shot selects exactly one action:

- `reference_media_to_video` takes 1–12 ordered references: at most 9 images, 3 videos, and 3
  standalone audio files, with at least one image or video.
- `first_last_frame_to_video` takes optional `first_frame` and `last_frame`. A non-first shot may
  use `first_frame: previous`; Creator binds the prior ordinary child's exact
  `continuation_frame`. An empty `{}` is valid T2VA. Because the scalar `previous` is reserved,
  `{asset: previous}` names a real file with that name.

Assembly uses either `audio: segments`, or `audio: master` plus `master_audio: <path>`. Master mode
replaces rather than mixes the generated segment soundtracks.

```sh
cozy video compose film.cozy-video.yaml \
  --h3 cozy/minimax-h3 --assembler cozy/video-assembly \
  --worker <rental-id> --out film.composition.json

cozy video submit film.cozy-video.yaml \
  --h3 cozy/minimax-h3 --assembler cozy/video-assembly \
  --worker <rental-id> --idempotency-key film-run-001
```

`compose` stages bounded immutable assets and records the path-free creative plan, but starts no
workflow. It prints a creative-plan digest; submitting that digest later reuses staged bytes without
reopening original paths. The caller retains the YAML as the editable source. Source bytes, creative
meaning, and deployment-resolved workflow execution have separate digests, so comments, path
spellings, and shot-ID edits do not change creative identity when their meaning and asset bytes stay
the same.

V1 deliberately has no includes, templates, variables, inheritance, arbitrary graph expressions,
transition system, retry fields, duration/step knobs, or model profile selection. FULL versus BAKED
is resolved from the exact endpoint release, outside the source.
