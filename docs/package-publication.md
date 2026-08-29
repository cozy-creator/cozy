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
credentials come from Cozy configuration or `TENSORHUB_TOKEN` and never enter project metadata.

The project also supplies `uv.lock` and `package.toml`. `uv.lock` preserves local installation
behavior; Tensorhub derives execution requirements from the built wheel and never dereferences
`[tool.uv.sources]`. Cozy skips `.env*`, credentials, VCS directories, virtual environments, caches,
editor state, bytecode, and build output. Modified and ordinary untracked files
are published normally.

Cozy runs `uv build --wheel` against the current tree. `uv` invokes the
project's declared PEP 517 backend; Cozy does not maintain another Python
package builder. Cozy verifies that the wheel name/version agrees with `[project]`, uploads it, and
Tensorhub independently verifies
that it is bounded `py3-none-any` content without native code, nested wheels,
credentials, unsafe paths, or executable `.pth` behavior.

Publication is one small release transaction:

1. Cozy opens the release with `POST .../begin`.
2. It registers safe relative source paths and receives release-scoped upload
   URLs for those files plus the wheel.
3. It uploads the files directly to object storage.
4. It calls `POST .../finalize` with an empty JSON object.

Cozy sends no digest/length manifest, descriptor, profile, GPU selection,
dependency list, model binding, or custom-wheel declaration. Tensorhub reads the
uploaded bytes, computes their hashes and lengths, inspects package metadata,
and derives compatible base worker images from its own inventory. Reopening a
committed release is an idempotent no-upload replay.

Finalize commits valid package custody even when no compatible base is active. In that case Cozy
reports a successful publication with zero qualified executions. Tensorhub later reconciles the
same release after base activation; it does not require another upload or package version. Serving
and rental selection remain closed until exact profile proof creates a qualified execution.

Cozy also supplies Tensorhub's internal audit text from the command and exact
package release. Publishers do not write an audit reason or release message.

Publication never builds an endpoint-specific Docker image and does not move
serving traffic.
