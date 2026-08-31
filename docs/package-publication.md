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

[tool.cozy]
organization = "paul"
```

The example publishes as `paul/marco-polo-package@1.0.0`. Cozy does not add a `v` prefix or accept
a second name/version on the command line. The configured `tensorhub_url` is the destination;
an enrolled machine authenticates automatically. An explicit operator credential may still come
from Cozy configuration or `TENSORHUB_TOKEN`; credentials never enter project metadata.

The project also supplies `uv.lock` and `package.toml`. Standard wheel `Requires-Dist` metadata is
the dependency authority, and the lock selects exact ordinary registry wheels.
For each runtime requirement backed by a local path or workspace member in `[tool.uv.sources]`, Cozy
recursively builds a separate non-editable wheel. Requested extras recursively activate their matching
`[project.optional-dependencies]` groups. Cycles, conflicting normalized names, incompatible
versions, more than 128 wheels, or more than 512 MiB of dependency wheels refuse before publication.
For ordinary registry requirements, Cozy runs `uv export --locked` and downloads exact
`py3-none-any` wheels from PyPI. Native-only, source-only, direct URL, VCS, and alternate-index
requirements are refused. Development dependency groups are ignored.
Source custody is limited to 20,000 files and 512 MiB total; each source file is limited to 64 MiB
and `uv.lock` to 16 MiB. The three required files must be non-empty.

Creator omits only the remote platform families that a rental may not replace: Python, Torch,
Runtime, TensorFS, and their platform closure. Ordinary libraries remain package-owned. Runtime
compares those platform requirements with the base Tensorhub selected before installing the package
environment.
Cozy refuses `.env*`, credentials, keys, bytecode, and model-weight files instead of silently
omitting them. It skips VCS directories, virtual environments, caches, editor state, and build
output. Modified and ordinary untracked files are published normally.

Cozy runs `uv build --wheel` against the current tree. `uv` invokes the
project's declared PEP 517 backend; Cozy does not maintain another Python
package builder. Cozy verifies that every local wheel's name and version agrees with its project and
the parent requirement. It also derives the canonical descriptor with the project's locked
`cozy-runtime describe` command. A build that repeatedly emits no output is stopped as stalled.

Publication is one small release transaction:

1. Cozy checks whether the exact release already exists; a committed replay stops here.
2. It builds the project wheel, dependency wheels, and canonical descriptor.
3. It opens or resumes the release with `POST /v1/packages/{org}/{name}/publish/{release}` and receives
   release-scoped upload URLs for the declared source paths, descriptor, project wheel, and dependency wheels.
4. It uploads the exact files directly to object storage.
5. It finalizes with `POST /v1/packages/{org}/{name}/publish/{release}/finalize` and an empty JSON object.

A committed replay skips every build and upload. At most 16 outstanding file uploads run concurrently.

Cozy sends no claimed wheel identity, digest/length manifest, profile, GPU selection, or model
binding. Tensorhub reads the uploaded bytes, computes their hashes and lengths, validates the
canonical descriptor shape, and inspects wheel metadata. Reopening a
committed release is an idempotent no-upload replay.

A pending retry keeps slots whose bytes Tensorhub already accepted and fills only missing slots
from the current tree. Changing the tree during a partial publication does not replace accepted
bytes; publish a new project version when the intended release contents changed.

Finalize commits valid package custody even when no base image is active. Publication neither
selects a base nor runs package code. Before a later rented execution spends money, Creator downloads
the exact immutable wheel set into disposable scratch and asks the installed Runtime to classify it
against Tensorhub's active WheelhouseManifests. The paid request carries only the sorted compatible
manifest digests. The selected worker still verifies the same wheels against its pinned base, installs
them offline, imports the package, and derives callable/model bindings.

Cozy also supplies Tensorhub's internal audit text from the command and exact
package release. Publishers do not write an audit reason or release message.

Publication never builds a package-specific Docker image and does not move
serving traffic.
