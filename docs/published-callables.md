# Published Python calls

A new callable package publication produces one source-preserving wheel before
Tensorhub seals its immutable environment. The App keeps the original functions
for direct serving. Public Python imports expose Runtime-generated managed caller
functions; ordinary helpers, globals, assets and result types remain intact.
The same wheel is installed as a serving package and as another package's dependency.
No worker import hook or mutation of a sealed environment is involved.

The generator records its ABI, the exact public interface and the hash of the
pre-overlay source wheel. That source hash is provenance, not an Environment ID.
Internal functions are excluded from dependency caller inventories. Runtime still
allows their existing same-revision self-calls. Re-capturing a published wheel for
a private script verifies the exact generated source and suffix before replacing
only that generated portion with the new capture's implementation identity.

Before accepting a published composition, Creator follows only dependency wheels
selected from the publisher's own Tensorhub organization index in the exact lock.
It verifies each selected SHA and version against the callee's committed project
wheel. The captured request includes every exact published environment, interface,
caller/callee binding and model default. Runtime owns preparation and child
execution on the selected machine; no client-side execution loop is needed after
acceptance. Package preparation does not download unrelated model weights.

Local dependency preparation also goes directly through Runtime's ordinary package
installer. It does not activate or replace the user's locally pinned package release.
Parent/child packages retain separate resolved environments, with uv sharing stored
wheel files on the machine.

Existing publications without generated caller exports remain directly runnable;
cross-package imports require a new caller-capable publication. There is no implicit
proxy compatibility path for old wheels. An implementation match requires exact
wheel bytes, not a name, a mutable latest release, or a matching interface alone.
