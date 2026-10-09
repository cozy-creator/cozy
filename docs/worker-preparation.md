# Package and model preparation

`cozy package install org/package` installs code and Python dependencies. It does
not inspect package model defaults or download weights. Package updates follow
the same rule; the obsolete `--no-model-download` switch has been removed.

Install package code on an existing rental by name or ID. As with a local
install, omitting `--version` selects the newest published release:

```sh
cozy package install paul/minimax-h3 --rental=kirukiru
cozy package install paul/minimax-h3 --rental=kirukiru --version 1.15.7
```

Rental installation leaves the local package inventory unchanged and sends no
model selections. It accepts published `org/name` packages only; `--editable`
and local directories are not supported by this published-release path. Installing
an editable package locally does not publish it. Use
`cozy run ./project/<function> --rental=kirukiru` for private local code execution
and its on-demand preparation.

Download a model directly into the rental's TensorFS store with a model selector:

```sh
cozy model download paul/minimax-h3#fp8-pruned --rental=kirukiru
cozy model download paul/minimax-h3@1.0.0-rc.2/fp8-pruned --rental=kirukiru
```

`#fp8-pruned` selects that lane from the newest eligible release containing it.
An explicit release/lane freezes that release. Models-only downloads install no
package and need no application slot or local destination. `#sha256:<digest>`
continues to name an exact retained checkpoint.

The rental must already have a local record (booting or ready). A pending
acquisition that has not produced a rental record cannot yet receive install
intents. Neither command creates or purchases a rental.

Both commands return a durable queued install ID. Creator resolves
and records the exact release and checkpoint at acceptance. A booting rental is
kept queued; the daemon delivers its accepted installs in order once its worker
is ready. The CLI may disconnect, and a restarted daemon resumes the same exact
selection without re-resolving latest or buying another rental.

The queue is managed in the background. Completion requires a verified Host receipt;
acceptance is not completion. `cozy model download --rental=NAME --await` waits for
that receipt, printing the machine's stage and bytes as it goes, and reports the elapsed
time; interrupting it stops only the wait. Failed boot, rental termination, or a changed
worker identity ends the install with a typed failure. Interrupted transfers
reuse the worker's retained download progress when Creator reconnects. The queue
stores exact logical selections, not presigned URLs. The worker refreshes download
authority at transfer time, so boot time cannot expire an accepted selection.

Package installation leaves model weights untouched. Model downloads do not
activate a serving placement or replace loaded packages. The former
`cozy rental prepare` command has been removed in favor of these two commands.

Standalone model downloads require the worker Host support introduced in
[Runtime PR #761](https://github.com/cozy-creator/cozy-runtime/pull/761).
Deploy that Host change in worker images before using this command; installing a
new Creator CLI alone does not update existing rentals.

Prewarming is optional. Inference prepares its exact model inputs before
admission. Private Python roots upload code directly to the worker, then acquire
only their selected root model inputs. Imported invocable defaults stay lazy:
Runtime selects and materializes a model when that child is actually called.

A local Hub control address can differ from its remotely reachable address. The CLI uses
its configured `tensorhub_url` for catalog and authorization requests. The selected Hub's
`GET /v1/execution-environment` supplies the remote `TENSORHUB_ORIGIN` from the Hub's
`server.public_origin` configuration. For a Hub running at `http://127.0.0.1:8819`, that
public origin can be its exposed HTTPS ngrok address. The remote machine reads package
releases, locked dependencies, model metadata and storage URLs there with an execution
grant bound to its pinned leaf key. It never treats the controller's loopback URL as its
own localhost.

A named rental remains controlled through its recorded machine identity and original
rental Hub. New runs, package installs and model downloads still use the CLI's selected
Hub. Every published request carries that source's own execution grant, even when the
same Hub rented the machine. A local unpublished run with no Hub model inputs can
still execute offline. A remote published source requires login to the selected Hub.

A bare remote `org/package/function` invocation names no release. The selected machine
provides its input schema and resolves the actual release when preparing execution. A
booting machine may use selected-Hub schema metadata for CLI input parsing and capacity
selection; that metadata never becomes a client-side version pin. A package missing from
the selected Hub fails even when this computer or the worker holds it from another Hub.
Only an explicit local source path takes the local-code upload path.
