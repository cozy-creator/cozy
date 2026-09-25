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
Use that implementation with the daemon's per-rental control hold. Do not spawn
the standalone developmenthold helper or stop the whole Creator daemon. Existing
ordinary images without this maintenance contract must give a clear unsupported
image error. New private rental defaults need an explicit maintenance-capable
image/access contract; shared worker image policy remains independent.

## Qualification

Use the ordinary Creator CLI against local Host fixtures for successful update,
repeated no-op, busy refusal, another rental remaining usable, invalid wheels,
rollback, client disconnect, daemon restart and retained TensorFS state. Biao's
observed failure was Runtime 0.18.2 against H3's >=0.18.3 requirement. The user ended
that rental; no new paid rental is authorized for this work.

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
The immutable 900-second rental idle deadline is neither extended nor disabled.
