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
hardware-qualified endpoint execution. Historical FULL rehearsals, community-curve artifacts, and
retired pre-hardcut lane names are forbidden. `accept.sh preflight` compares Creator's verified Tensorhub
control snapshot against the authorized endpoint-execution digest and complete model-root set.

The stages are deliberately progressive:

1. `single` — ordinary Creator Ref2VA invocation and local output download.
2. `two-recover` — real two-shot source, exact continuation binding, Creator SIGKILL/restart,
   idempotent resubmission, terminal follow, and verified local workflow download.
3. `cancel-two` — a distinct workflow canceled while shot two is active; no assembly child may
   appear afterward.
4. `eight` — enabled only after the previous automated bundles and manual review receipt pass.

No stage has an elapsed-time failure verdict. Status polling is sampling; Creator settles stalls
from worker liveness and measured no-progress reports. Manual watch/listen remains mandatory.

Copy the YAML fixtures before editing prompts or asset paths. Set the variables documented by
`accept.sh usage`; every output directory and idempotency key must be fresh for a distinct run.
Use `verify.py bundle` after every download and fill `manual-review.template.json` from actual
watch/listen inspection. The existing Serverless `video-assembly/live.py` remains the one small
deterministic synthetic assembly check; this bundle adds no mocked suite.
