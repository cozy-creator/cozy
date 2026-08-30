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

The project also supplies `uv.lock` and `package.toml`. `uv.lock` preserves local installation
behavior, while standard wheel `Requires-Dist` metadata remains production dependency authority.
For each runtime requirement backed by a local path or workspace member in `[tool.uv.sources]`, Cozy
recursively builds a separate non-editable wheel. Requested extras recursively activate their matching
`[project.optional-dependencies]` groups. Cycles, conflicting normalized names, incompatible
versions, more than 32 wheels, or more than 512 MiB of dependency wheels refuse before publication.
Direct URL and VCS requirements are not accepted. Development dependency groups are ignored.
Source custody is limited to 20,000 files and 512 MiB total; each source file is limited to 64 MiB
and `uv.lock` to 16 MiB. The three required files must be non-empty.

Creator does not infer base ownership from distribution names. It uploads every referenced local
candidate; Tensorhub checks standard requirements against each exact base worker image, chooses its
base copy whenever compatible, and leaves that uploaded candidate out of the overlay. Tensorhub also
resolves non-base indexed requirements and mirrors exact wheels. An overlay never shadows the base.
Cozy refuses `.env*`, credentials, keys, bytecode, and model-weight files instead of silently
omitting them. It skips VCS directories, virtual environments, caches, editor state, and build
output. Modified and ordinary untracked files are published normally.

Cozy runs `uv build --wheel` against the current tree. `uv` invokes the
project's declared PEP 517 backend; Cozy does not maintain another Python
package builder. Cozy verifies that every local wheel's name and version agrees with its project and
the parent requirement. Tensorhub independently inspects the uploaded bytes and decides profile
compatibility. A build that repeatedly emits no output is stopped as stalled.

Publication is one small release transaction:

1. Cozy opens the release with `POST /v1/packages/{org}/{name}/publish/{release}`.
2. It registers safe relative source paths and local dependency-wheel basenames with `POST` to that
   release's `/uploads` route, then receives
   release-scoped upload URLs for those files plus the project wheel.
3. It uploads the exact files and separate wheels directly to object storage.
4. It finalizes with `POST /v1/packages/{org}/{name}/publish/{release}/finalize` and an empty JSON object.

The open call happens before wheel construction. A committed replay therefore skips every build
and upload. At most 16 outstanding file uploads run concurrently.

Cozy sends no claimed wheel identity, digest/length manifest, descriptor, profile, GPU selection,
model binding, or resolved dependency set. Tensorhub reads the uploaded bytes, computes their hashes
and lengths, inspects package metadata,
and derives compatible base worker images from its own inventory. Reopening a
committed release is an idempotent no-upload replay.

A pending retry keeps slots whose bytes Tensorhub already accepted and fills only missing slots
from the current tree. Changing the tree during a partial publication does not replace accepted
bytes; publish a new project version when the intended release contents changed.

Finalize commits valid package custody even when no compatible base is active. In that case Cozy
reports a successful publication with zero qualified executions. Tensorhub later reconciles the
same release after base activation; it does not require another upload or package version. Serving
and rental selection remain closed until exact profile proof creates a qualified execution.

Cozy also supplies Tensorhub's internal audit text from the command and exact
package release. Publishers do not write an audit reason or release message.

Publication never builds a package-specific Docker image and does not move
serving traffic.
