# H3 long-form paid acceptance

This bundle drives the real Creator product surfaces. It never rents capacity implicitly and it
contains no model root, lane, provider, datacenter, cache, or artifact path. Before using it, the
operator explicitly runs:

```sh
cozy rent <org/endpoint/v1/reference_media_to_video> \
  --accelerator '<provider-neutral GPU SKU>' \
  --idempotency-key <paid-rental-key> --reason '<why this proof is authorized>'
```

Do not run that command until Tensorhub has published the expected `fp8-adaln-pruned` snapshot and
hardware-qualified endpoint execution. Unpublished or superseded execution artifacts are forbidden.
`accept.sh preflight` first runs the read-only
`cozy rent probe` ClaimAck, then compares Creator's persisted live GPU readback and verified
Tensorhub control snapshot against the exact requested endpoint, authorized endpoint-execution
digest, and complete model-root set. The probe identifies the worker but invokes no model.

The stages are deliberately progressive:

1. `single` — ordinary Creator Ref2VA invocation and local output download.
2. `two-recover` — compose once, submit the retained creative-plan digest, verify the exact
   continuation binding, SIGKILL/restart Creator, replay that same digest and key, follow the
   terminal, and download the workflow locally.
3. `cancel-two` — a distinct workflow canceled while shot one is active; neither shot two nor
   assembly may appear afterward.
4. `eight` — enabled only after the previous automated bundles and manual review receipt pass.

No stage has an elapsed-time failure verdict. Status polling is sampling; Creator settles stalls
from worker liveness and measured no-progress reports. Manual watch/listen remains mandatory.

The paid commands are explicit and progressive:

```sh
export RENTAL_ID=<rental-id>
export H3_ENDPOINT=cozy/minimax-h3
export GPU_SKU='<exact provider-neutral GPU SKU>'
export EXPECTED_ENDPOINT_EXECUTION=sha256:<64-hex>
export EXPECTED_MODEL_ROOTS='<comma-separated sha256 roots>'

export REF_IMAGE=<image> REF_VIDEO=<video> REF_AUDIO=<audio> SINGLE_KEY=<fresh-key>
proofs/h3-long-form/accept.sh single proof-single

export SINGLE_RECEIPT_DIR=proof-single
proofs/h3-long-form/accept.sh two-recover two-shot.cozy-video.yaml <fresh-key> proof-two

export TWO_RECEIPT_DIR=proof-two
proofs/h3-long-form/accept.sh cancel-two two-shot.cozy-video.yaml <fresh-key> proof-cancel
```

Those three commands must finish, and single/two must be watched and listened to, before `eight`:

```sh
export CANCEL_RECEIPT_DIR=proof-cancel
proofs/h3-long-form/accept.sh eight eight-shot.cozy-video.yaml <fresh-key> \
  manual-review.json proof-eight
```

Copy the YAML fixtures before editing prompts or asset paths. Set the variables documented by
`accept.sh usage`; every output directory and idempotency key must be fresh for a distinct run.
Each workflow stage retains `composition.json`, so recovery never reopens the YAML or its asset
paths. Automated receipts compare video and audio stream endpoints independently as well as frame
count, codecs, signal integrity, segment order, and continuation binding.
Bundle-index digests protect exact downloaded files; manual-review digests canonicalize parsed JSON,
so whitespace and object-key order never invalidate a human verdict.

Copy `fixtures/manual-review.template.json` and bind each manual verdict to the corresponding
semantic verification-document digest:

```sh
python3 proofs/h3-long-form/verify.py digest <run-dir>/verification.json
```

The eight-shot prerequisite gate requires digest-bound single/two watch-listen review and the
two-shot transition decision. After the eight-shot run, watch/listen the final output, decide each
of its seven transitions separately, fill the `eight` section, and close acceptance with:

```sh
proofs/h3-long-form/accept.sh finalize-eight <eight-run-dir> <manual-review.json>
```

That writes `manual-acceptance.json` bound to the prerequisite gate, final automated verification,
and manual document. The existing Serverless `video-assembly/live.py` remains the one small
deterministic synthetic assembly check; this bundle adds no mocked suite.
