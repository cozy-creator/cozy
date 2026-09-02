# Package publication

Cozy publishes the current project tree. A Git repository, commit, or clean
working tree is not required.

```sh
cozy package publish
```

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
lock rows — `(name, version, url, sha256, size)` — for Tensorhub to fetch itself under the
same pinned discipline; the bytes never move through this machine (cl-078). Native-only,
source-only, direct URL, VCS, and alternate-index
requirements are refused. Development dependency groups are ignored.
Source custody is limited to 20,000 files and 512 MiB total; each source file is limited to 64 MiB
and `uv.lock` to 16 MiB. The three required files must be non-empty.

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
The only publish-time metadata derivation is the project's locked `cozy-runtime describe`;
its canonical output is published as `metadata/package-interface.json`. Publication derives
no checkpoint evidence, code topology, tensor requirements, or compatibility cache.

The uploaded ordinary file tree contains the source paths, the project wheel at
`artifacts/project/<wheel>`, genuinely bundled dependency wheels at
`artifacts/dependencies/<wheel>`, and `metadata/package-interface.json`. Registry wheels are
external locked environment facts and are never re-uploaded.

Publication is one bounded digest-declared session:

1. Cozy checks whether the exact release already exists; a committed replay stops here.
2. It builds the project and bundled dependency wheels, derives the PackageInterface, and hashes
   every ordinary file locally.
3. `POST /v1/packages/{org}/{name}/publish/{release}` carries `{files:[{path,digest,length}]}`.
   Tensorhub journals the declaration under a `publication_id` and answers each file from the
   store's own HEAD: already present, or one checksum-pinned single-object PUT.
4. Cozy PUTs only the absent subjects; a 412 stands as success (the key already holds these
   exact bytes), so an unchanged republish uploads nothing.
5. It finalizes with `{publication_id,registry:[...]}`. Tensorhub streams and re-hashes the
   session's stored files, fetches registry wheels, validates the PackageInterface and wheel
   environment, and atomically commits the immutable release and ordinary file rows in Postgres.
   There is no package manifest, aggregate release digest, or committed marker object.

A committed replay skips every build and upload. At most 16 outstanding file uploads run concurrently.

The declared digests are claims, never authority: Tensorhub's own hash of the stored bytes
remains identity, and wrong bytes at any declared digest require a new release id. If a pod
later derives a different PackageInterface than the committed one, its typed refusal
(`package_prepare_interface_disagrees`) is relayed to
`POST /v1/packages/{org}/{name}/releases/{release}/defects` with the rental's signed
delegation as the chain of authority, and the release is tombstoned until a corrected one
is published.

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
