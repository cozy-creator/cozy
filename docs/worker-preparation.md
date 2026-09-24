# Package and model preparation

`cozy package install org/package` installs code and Python dependencies. It does
not inspect package model defaults or download weights. Package updates follow
the same rule; the obsolete `--no-model-download` switch has been removed.

Prewarm a rental by name or ID with an exact package release:

```sh
cozy rental prepare bisco paul/minimax-h3 --version 1.15.7
```

That command also installs code only. Explicit model selections opt into weights:

```sh
cozy rental prepare bisco paul/minimax-h3 --version 1.15.7 \
  --model cut_segment_turbo.models.base_model=paul/minimax-h3@1.0.0-rc.2/fp8-pruned
```

Repeat `--model` for each required slot. Creator resolves the release/lane to an
exact manifest and forwards that selection to the worker. Completion means the
Host returned a verified `PREPARED` receipt. Sending a desired state is not
completion. Preparation does not activate a serving placement or replace other
loaded packages. Interrupted preparation can be repeated against the same
worker; the worker owns retained download progress and installed environments.

Prewarming is optional. Inference prepares its exact model inputs before
admission. Private Python roots upload code directly to the worker, then acquire
only their selected root model inputs. Imported invocable defaults stay lazy:
Runtime selects and materializes a model when that child is actually called.

A local Hub control address can differ from its remotely reachable byte address.
The authenticated rental `image-inventory` response supplies `public_origin` for
private child captures. Captures verify anonymous access to each exact checkpoint
at that origin; no client credential or private source is sent there. Creator
continues using the configured local Hub for control requests. Public HTTPS
origin validation remains in place, so a loopback URL cannot become a remote
worker's download address by accident.
