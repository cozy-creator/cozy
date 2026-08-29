# Package publication

Cozy publishes the current project tree. A Git repository, commit, or clean
working tree is not required.

```sh
cozy package publish org/package \
  --release 1.0.0 \
  --dir .
```

The project supplies `pyproject.toml`, `uv.lock`, and `package.toml`. Cozy
skips `.env*`, credentials, VCS directories, virtual environments, caches,
editor state, bytecode, and build output. Modified and ordinary untracked files
are published normally.

Cozy runs `uv build --wheel` against that current tree. `uv` invokes the
project's declared PEP 517 backend; Cozy does not maintain another Python
package builder. Cozy uploads the one resulting wheel, and Tensorhub verifies
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

Cozy also supplies Tensorhub's internal audit text from the command and exact
package release. Publishers do not write an audit reason or release message.

Publication never builds an endpoint-specific Docker image and does not move
serving traffic.
