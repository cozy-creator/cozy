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

ABI 8 publication requires an authored `cozy-runtime>=0.18.21` lower bound and an
actual locked Runtime at least 0.18.21. A newer lock alone cannot hide an older
advertised floor. Publication refuses before upload with the exact required floor;
it does not silently rewrite an author's lock. The same compiled SDK wheel can be
captured again by a private script after its generated source is verified.

Real local Creator CLI qualification used two test publications in the local Hub,
with both `COZY_HOME` and `TENSORFS_HOME` isolated. The parent returned 21 from its
locked child 0.1.0 and preserved the child's ordinary helper result 12. Direct
serving from that same 0.1.0 wheel returned 24. After child 0.1.1 changed its
multiplier, same-package managed calls returned 36 while the unchanged parent's
exact dependency still returned 21. Switching those retained package placements
also exercised the Runtime readiness guard: a previous warm placement must not
allow an offer before the selected replacement is dispatchable.

The fixture publications used development Runtime code reporting 0.18.20 before
this early floor validation was added; they are test artifacts, not a supported
production release cohort. Runtime 0.18.21 and the updated first-party package
locks are mandatory deployment gates.
