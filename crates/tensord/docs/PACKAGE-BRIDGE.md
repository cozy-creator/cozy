# Package bridge

`python/tensord_client` describes packages and builds immutable uv environment generations. Every
call runs on a pool executor ([GPU service](GPU-SERVICE.md)); the journal side is in
`DURABLE-EXECUTION.md`.

## Generations (`environments.rs`; local closures by `captured_packages.py`)

Layout: `<generations>/<32-hex>/` with `env/`, `.hold` and `generation.json`, the typed
`catalog::Generation` (`identity`, `package`, `version`, `application`, `python`, every installed
distribution in `dependencies`, `interface`, `source_digest`, `callees`, `sdk_fallback`,
`runtime_build`). `generation.json` is published atomically last. A generation is never updated.
`catalog.rs` resolves one only under a shared `.hold` flock; `reclaim.rs` collects it.

A release's environment (`published.rs`, from its lock) and a local package's (`api/install.rs`)
take one path once their own closure is in place (`environments.rs`):
- The machine's own Runtime/TensorFS pair replaces the closure's, every other version constrained,
  where `uv pip check` admits it. Else the closure's own pair is restored and `sdk_fallback` says
  why; every run of it logs that warning. A development Runtime an update named (`--runtime-wheel`)
  runs whatever the package's bounds on the pair, and `runtime_build` names it.
- The client wheel is installed under the same constraints. Then `uv pip check`'s findings
  (each missed requirement, but a pair bound the development Runtime runs regardless) are
  recorded in `findings`: never a refusal (version skew is allowed); the install log and every
  run say them. A wheel tagged for another platform is no finding.
- The client describes the environment inside it with `python -I -S` (no `.pth` execution): its
  source digest, its callees, and a local package's interface (a live project's from its directory).

`capture` (`python -m tensord_client.installer`, called by `api/install.rs`) builds a local
package's own closure, nothing more:
- Python comes from `uv python find` and must match the captured `python_requires`/`python_version`.
- Locked source: `uv sync --frozen --no-dev --no-editable`, its own `[tool.uv]` settings holding.
- Hashed requirements: `--require-hashes`, then the wheels with `--no-deps`.
- Anything else: `uv pip install` of the project and/or wheels, the machine's pair offered.
- A locked source mixed with wheels or requirements is refused.
- `--sdk-wheel` names the machine's pair: a lock's own pair is left out (the machine installs
  its own next); `--restore` installs the closure's own pair where the machine's cannot stand.
- A live project installs editable in its own environment, kept while its lock stands (`.ready`
  holds the machine's `sdk_fallback` and `runtime_build`).
- Failure prints `{kind: "install_failed", code, detail}`. `install.rs` passes the helper's
  stderr to the machine log and quotes its last lines (credentials masked) in the refusal when
  the failure is an operation's (`package_dependency_operation_failed`) or untyped.

## Calls between packages

A generation also records `callees`: the other installed distributions with a statically
described `cozy.application`, their application, interface, package identity and source digest.
Their child runs use the caller's immutable environment and the callee's own App. A callee
can call its own internal exports; another package can call only its public exports. Memoized
calls use the callee's source digest.

A local-source manifest may carry an optional `callees` map from normalized distribution names
to package identities. It may carry a source archive, or an immutable root wheel together with
dependency wheels. A published release's map is its lock: every row its Hub publishes is
`<release org>/<row name>` (a Hub lock pins only its own org). A wheel's URL never names a
package; anything unmapped, and older records without a package identity, read as
`local/<normalized-distribution>`. Description
reads metadata and AST inside the environment without importing package code.

Model slots and resolution-cache entries belong to the callee package. The caller's model
choices do not override another package's slots; a choice addressed to a callee slot names it. GPU
execution uses the callee's interface; import-only parents are keyed by generation and App.
These are additive generation and manifest fields; the machine API is unchanged.
An export registered by distinct Apps is refused only when that ambiguous call is made;
other calls and same-App aliases continue to work.
