# Private rental Runtime updates

`cozy rental update NAME` updates worker software on an existing private rental.
Package installation remains separate. Updating a package can raise its Runtime
requirement; that requirement must be checked against the worker's observed
distribution before importing the package.

Creator's existing daemon owns the operation and the rental's dispatch hold. The
short-lived CLI starts or observes it. The Host owns replacement of its Runtime
process and the worker's installed distributions. TensorFS state stays outside
the replaceable environment. Creator does not run inside the worker.

## Maintenance sequence

1. Resolve the attached rental and pin its existing worker, boot, certificate and
   developer maintenance endpoint. Refuse a missing or ended rental.
2. Resolve latest stable published native Runtime and TensorFS wheels directly from
   PyPI against the worker's actual interpreter and platform. TensorFS must satisfy
   the selected Runtime wheel's dependency metadata. Verify archive identities,
   lengths and SHA256 digests. No registered image or Hub update approval is needed.
   Keep heavy image distributions unchanged; the worker performs its existing
   offline dependency preflight before replacement.
3. Record the exact selected pair and maintenance operation in Creator's SQLite.
   Withdraw only this rental from new preparation and dispatch. Active requests
   make the update wait or refuse; they are never interrupted.
4. Transfer the selected wheels, claim the signed idle snapshot with admission
   held closed, and let the existing Host updater validate dependencies, native
   imports and the unchanged platform. The Host cooperatively replaces Runtime.
5. Require actual selected-version readback, a newer control stream, and the same
   retained boot and TLS identity before reopening dispatch. Failed candidates
   roll back to the previous healthy pair.

An interrupted CLI does not cancel the daemon operation. A daemon restart sees
the durable maintenance row and keeps dispatch closed while reconciling the Host
selection. Unknown remote completion is neither success nor proof of rollback.

## Automatic compatibility repair

Before preparation, compare authored `Requires-Dist` constraints for worker-owned
Runtime/TensorFS with their actual installed versions. Older workers need this
client preflight because they cannot emit the new dependency refusal. Runtime
also validates the dependency boundary before import and returns typed facts.

One compatible update may repair an unsent request and retry preparation with
the same request ID. Never parse a Python traceback to decide to update. Never
repeat an update indefinitely, and never duplicate an accepted submission.
Unsupported heavy-platform changes require a compatible image; this operation
does not purchase or replace a rental.

## Initial maintenance transport

The existing developer Host wrapper already provides exact-wheel validation,
cooperative replacement, rollback, retained readiness and an operator SSH path.
Use that implementation with the daemon's per-rental control hold; never stop the
whole Creator daemon. Existing
ordinary images without this maintenance contract must give a clear unsupported
image error. New private rental defaults need an explicit maintenance-capable
image/access contract; shared worker image policy remains independent.

## Qualification

Use the ordinary Creator CLI against local Host fixtures for successful update,
repeated no-op, busy refusal, another rental remaining usable, invalid wheels,
rollback, client disconnect, daemon restart and retained TensorFS state. Biao's
observed failure was Runtime 0.18.2 against H3's >=0.18.3 requirement. Real rental
qualification uses an explicitly owned rental and the ordinary Creator CLI.

## Independent release selection

Image releases install a tested starting environment. They do not authorize or
select later operator-requested Runtime updates. Creator resolves the public
`cozy-runtime` and `tensorfs` projects for the observed worker platform, excludes
prereleases, yanked files and universal-only substitutes, and never silently
downgrades an installed distribution. It records the selected exact wheel URLs,
versions, lengths and hashes before attempting replacement. Reconciliation uses
that frozen selection even if a newer release appears meanwhile.

The managed worker interpreter is `/opt/cozy/python/bin/python3`; maintenance
commands do not depend on SSH's interactive PATH. The current wheel-pair symlink
is a rollback cache, not a virtual environment. Worker dependency validation,
native imports and post-update health remain authoritative. An incompatible
heavy platform is an actionable preflight refusal, not permission to replace it.
An update is idle time: it neither resets nor disables the rental's 900-second idle clock.

The CUDA image installs `uv` at `/usr/local/bin/uv`. Its existing updater invokes
Python and supervisorctl with absolute paths, so older images with the original
`/usr/local/bin:/usr/bin:/bin` child PATH can use this hot-update path without
replacing the updater. Supervisord's host/updater processes inherit the managed
Python PATH from the image entrypoint; they do not inherit the SSH session PATH.

## Development candidate wheel

The owner of a maintenance-capable development rental can run:

```sh
cozy rental update NAME --runtime-wheel /absolute/path/cozy_runtime-VERSION-PLATFORM.whl
```

Creator snapshots the bounded local file into its private staging directory and
records its SHA256 and length atomically with the maintenance operation. The
candidate must be a native `cozy-runtime` wheel compatible with the observed
worker interpreter/platform and contain matching distribution metadata. Use a
unique newer version, including the repository builder's content-addressed local
version; replacing bytes under the currently installed version is refused.
TensorFS comes from the public index by default. To retain a tested development
TensorFS build, also pass `--tensorfs-wheel /absolute/path/tensorfs-VERSION-PLATFORM.whl`.
This option requires `--runtime-wheel`; both files are frozen and recovered as one
operation without contacting PyPI. TensorFS must satisfy the selected Runtime
requirements, support the worker platform, and must not downgrade the installed
version. Reusing the same TensorFS version is allowed. Other installed platform dependencies remain unchanged and are
checked by the existing offline worker preflight.

The same authenticated owner-only local API, development-rental ownership check,
signed idle hold, SFTP transfer, wheel hashes, guarded restart, durable recovery
and rollback apply. This option neither registers an image nor publishes bytes.
It is explicit development authority, never selected by automatic repair.

If the CLI disconnects, run `cozy rental update NAME` again to resume the recorded
operation without reopening the build output. Repeating the local option while
an operation is active must name the same filename and bytes; a different
candidate cannot replace an unfinished operation. A missing original build file
does not prevent resume. After success, use a new version for another candidate;
the default command continues to select public releases without downgrading.

## Execution protocol changes and maintenance

Runtime repair has a separate admission decision from execution. A Creator that
requires execution wire 62 can inspect and maintain a worker on wire 60. The
current maintenance lane accepts the known Claim/closed-snapshot contract from
wire 60 through Creator's current wire version, requires the rental keepalive
capability, and signs a Claim using the peer's supported intersection. It never
sends SnapshotAck, desired state, preparations, or attempt offers. Certificate,
worker/boot identity, owner signature, durable local custody, held outcomes, and
the updater's independent active-work restart guard remain mandatory. A failed
execution probe is not evidence that a worker is idle.

The existing protocol probe reports the intersection of Host and installed
Runtime. Before applying any wheel, Creator conservatively requires this range
to reach the candidate Runtime's declared minimum. A Host on wire 60 cannot run
a Runtime requiring wire 62. Installing Runtime/TensorFS wheels does not upgrade
the Host binary; the existing guardian only restarts that binary. A range
refusal leaves both components unchanged and requires a compatible worker image.
Even a newer Host hidden behind an older Runtime's intersection is conservatively
refused until separate Host capability reporting is available.

### Required follow-up: independent Host and Runtime recovery

A genuinely independent maintenance protocol must live in the supervisor, outside
both replaceable Host and Runtime processes. It must report its own version and
each component's independent range, authenticate the existing owner and pinned
worker lifetime, and expose only inspect/stage/apply/reconcile operations. It must
remain usable when Host and Runtime have no execution-protocol overlap.

Extending the existing guardian requires immutable, digest-verified Host binary
and wheel selections as one durable operation; compatibility and platform checks
before stop; independent Host-ledger and Runtime active-work fences; dispatch
withdrawal; and retained identity, workspace, deadline, and rollback selection.
The guardian must restart and validate the entire selected pair, roll back both
components on a failed health check, and retain ambiguous operations for explicit
reconciliation. A new protocol must not grant execution authority or bypass idle
shutdown. The current wheel-only API does not implement this Host replacement
contract. Qualify disconnect/restart, active work, wrong owner, failed Host start,
failed Runtime start, and full rollback before enabling Host updates.
