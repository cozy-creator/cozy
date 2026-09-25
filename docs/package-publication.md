# Package publication

Cozy publishes the current project tree. A Git repository, commit, or clean
working tree is not required.

```sh
cozy package publish
```

To publish an existing wheel set from CI or a platform build matrix, repeat
`--wheel` for each artifact:

```sh
cozy package publish --wheel dist/example-1.0.0-cp312-cp312-manylinux_2_28_x86_64.whl \
  --wheel dist/example-1.0.0-cp313-cp313-manylinux_2_28_x86_64.whl
```

Supplying wheels skips the project wheel build. Cozy still builds the standard
sdist from the current source tree and uploads its exact Runtime metadata.
Every wheel must match the project's name, version, and application entry point.
Tensorhub verifies each wheel's archive, RECORD, tags and common dependency
metadata. It retains all variants in the package index and selects a compatible
artifact for the requested executor. Unsupported Python/ABI/platform combinations
refuse before installation.

`pyproject.toml` is the publication authority:

```toml
[project]
name = "marco-polo-package"
version = "1.0.0"
```

If the authenticated Tensorhub account is `paul`, the example publishes as
`paul/marco-polo-package@1.0.0`. Cozy does not add a `v` prefix or accept a second owner,
name, or version on the command line. The configured `tensorhub_url` is the destination;
an enrolled machine authenticates automatically. Credentials never enter project metadata.

The project also supplies `uv.lock` and `package.toml`. Standard wheel `Requires-Dist` metadata is
the dependency authority, and the lock selects exact ordinary registry wheels.
For each runtime requirement backed by a local path or workspace member in `[tool.uv.sources]`, Cozy
recursively builds a separate non-editable wheel. Requested extras recursively activate their matching
`[project.optional-dependencies]` groups. Cycles, conflicting normalized names, incompatible
versions, more than 128 wheels, or more than 512 MiB of dependency wheels refuse before publication.
For ordinary registry requirements, Cozy runs `uv export --locked` and sends the resulting
lock rows — `(name, version, url, sha256, size)` — for Tensorhub to validate as metadata.
Tensorhub does not download registry dependencies during publication. Source-only,
direct URL, VCS, and alternate-index
requirements are refused. Development dependency groups are ignored.
Source custody is limited to 20,000 files and 512 MiB total; ordinary source files are limited to 64 MiB,
prebuilt dependency wheels to 512 MiB, and `uv.lock` to 16 MiB. The three required files must be non-empty.

Creator omits only the remote platform families that a rental may not replace: Python, Torch,
Runtime, TensorFS, and the small declared native platform closure shared by the supported Torch
bases. Ordinary pure-Python libraries remain package-owned. Runtime compares those platform
requirements with the base Tensorhub selected before installing the package environment.

For local installation, Creator creates a normal uv venv using the release's Python requirement.
It exports the already-committed `uv.lock` with `--frozen` because author-local source paths no
longer exist on the consumer; those distributions come from their exact published wheels. Registry
rows install with their lock hashes, then the project/custom wheels install without builds, and
`uv pip check` joins the complete environment back to the wheel requirements. Publication itself
already ran `uv export --locked`, so a mismatched project and lock never became a release.
Cozy refuses `.env*`, credentials, keys, bytecode, and model-weight files instead of silently
omitting them. It skips VCS directories, virtual environments, caches, editor state, and build
output. Modified and ordinary untracked files are published normally.

Cozy runs `uv build --wheel` against the current tree. `uv` invokes the project's declared
PEP 517 backend; Cozy does not maintain another Python package builder. Cozy verifies that
every bundled local wheel's name and version agrees with its project and parent requirement.
Publication derives no metadata. The tree's committed `metadata/package-interface.json` is the
PackageInterface Tensorhub accepts as truth (decision #713); the package's own tooling writes
it. Right before upload Cozy pre-flights it: this host's `cozy-runtime describe` — a release
at or above `hostruntime.Floor`, whose reading is static — parses the tree and imports
nothing, so no package code runs on the publisher's host and no environment is built for the
question. A committed file that differs from that reading, or is absent, is refused as
`package_publish.interface_stale`, naming the first difference; on a match the committed file
uploads unchanged. The same static reading, by the same host tool, is what install compares
with the committed release and what a job's descriptor id is read from (cl-175). Publication
derives no checkpoint evidence, code topology, tensor requirements, or compatibility cache.

The uploaded artifacts are a standard sdist at `artifacts/source/<sdist>.tar.gz`,
one or more project wheels at `artifacts/project/<wheel>`, bundled dependency wheels at
`artifacts/dependencies/<wheel>`, and `metadata/package-interface.json`. Registry wheels are
external locked environment facts and are never re-uploaded. `package.toml`,
`pyproject.toml`, and `uv.lock` are also uploaded as exact runtime documents.

Publication is one bounded digest-declared session:

1. Cozy checks whether the exact release already exists; a committed replay stops here.
2. It builds the project and bundled dependency wheels, pre-flights the committed
   PackageInterface, and hashes every ordinary file locally.
3. `POST /v1/packages/{org}/{name}/publish/{release}` carries `{files:[{path,digest,length}]}`.
   Tensorhub journals the declaration under a `publication_id` and answers each file from the
   store's own HEAD: already present, or one checksum-pinned single-object PUT.
4. Cozy PUTs only the absent subjects; a 412 stands as success (the key already holds these
   exact bytes), so an unchanged republish uploads nothing.
5. It finalizes with `{publication_id,registry:[...]}`. Tensorhub streams and re-hashes the
   session's stored files, validates registry metadata, the PackageInterface and wheel
   environment, and atomically commits the immutable release and ordinary file rows in Postgres.
   There is no package manifest, aggregate release digest, or committed marker object.

A committed replay skips every build and upload. At most 16 outstanding file uploads run concurrently.

The declared digests are claims, never authority: Tensorhub's own hash of the stored bytes
remains identity, and wrong bytes at any declared digest require a new release id. If a pod
later derives a different PackageInterface than the committed one, its typed refusal
(`package_prepare_interface_disagrees`) is relayed to
`POST /v1/packages/{org}/{name}/releases/{release}/defects`, authorized by the caller's
OWNERSHIP of the named rental, and the release is tombstoned until a corrected one is
published. The signed-delegation chain that used to prove the rental had actually
downloaded the bytes is deleted (owner ruling 2026-09-03), so the report's provenance is
no longer proven.

Finalize commits valid package custody even when no base image is active. Publication neither
selects a base nor runs package code. A later paid request names only a rental SKU and the
authentication facts needed to attach the worker; it does not disclose a package or a base
inventory. After Creator connects, it delegates the exact immutable wheel set directly to the
worker. Runtime observes the worker environment, refuses protected platform-package conflicts,
installs the package environment offline, imports the package, and derives callable/model bindings.

Cozy also supplies Tensorhub's internal audit text from the command and exact
package release. Publishers do not write an audit reason or release message.

Publication never builds a package-specific Docker image and does not move
serving traffic.

## Unpublished packages

An unpublished package runs from captured source without creating a Tensorhub package
release. A local script is the single-file form. Editable describes the dependency
installation mode, not a separate execution or privacy mode.

Worker protocol `Private*` names, stable `private_*` error codes, persisted capture markers,
and historical database fields retain their serialized spellings for existing workers
and requests. New code and explanations use the publication terminology. Private rentals
and access controls continue to mean private access.

## Selected Python interpreter

Runtime owns the `cozy.python-interpreters/1` response: its supported execution
window and measured standard CPython executors (executable, exact patch, ABI).
Creator reads this ephemeral response; it is neither persisted as a document
nor assigned a content digest. Public release finalization sends the selected
`python_version` separately from the author's broad `Requires-Python` metadata.
Hub retains that selection in the release environment and returns it in download
plans and preparation facts. Before renting, Creator requires the captured
version and ABI to appear in the SKU's measured executor inventory.

The production image may advertise only Python 3.12. A local 3.13 or 3.14
capture cannot rent that image; additional executors must already be present
and advertised. Interface generator ABI 7 declares Python >=3.12, independently
of the Runtime execution window.

### Selecting a development Tensorhub index

Keep ordinary uv metadata pointed at the canonical organization index:

```toml
[tool.cozy]
organization = "paul"

[tool.uv.sources]
qwen-image-2 = { index = "tensorhub-paul" }

[[tool.uv.index]]
name = "tensorhub-paul"
url = "https://tensorhub.com/v1/index/paul/simple/"
explicit = true
```

For a development Hub, explicitly author and review its lock before publication:

```sh
uv lock --index tensorhub-paul=http://127.0.0.1:8819/v1/index/paul/simple/
cozy package publish --tensorhub=http://127.0.0.1:8819
```

Standard uv records the selected index and exact artifact URLs/hashes in `uv.lock`;
the pyproject stays canonical, while the lock honestly identifies the selected Hub.
Publication passes the matching named-index override to `uv export --locked` and
refuses a different-Hub or stale lock. It never silently relocks or rewrites either
file. Moving to production requires explicitly locking against production and
reviewing that change. No separate Cozy lock command is needed.

Only explicitly named canonical Tensorhub indexes for the project's own declared
organization are overridden. PyPI, other organizations, and third-party indexes
retain their existing behavior and validation. The same override is used when
Cozy locks its owned editable-package or client-script capture; authored files
remain unchanged. Index selection does not expand the child environment allowlist
or forward Hub credentials to uv.
