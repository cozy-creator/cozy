# Package publication

Creator publishes package code. It never builds a package Docker image or selects a GPU.
It runs the project's declared Python wheel backend only in private publication staging.

## Source release

The source directory does not need to be a Git checkout or have a clean working tree. Its current
modified and ordinary untracked files are publication inputs. It contains:

- static project metadata and dependencies in `pyproject.toml`;
- an exact `uv.lock`;
- author intent in `package.toml`;
- pure Python project files.

`package.toml` is the only author configuration. `package.release.json` and
`package.evaluated-config.json` are retired, as is committed `package.descriptor.json`.
Native files, build recipes, nested wheels, model weights, credentials, symlinks, and
unsafe paths refuse. VCS metadata, virtual environments, caches, editor state, and prior build
output are excluded from the private snapshot.

```sh
cozy package publish org/package \
  --release 1.0.0 \
  --reason "initial release"
```

Creator snapshots the source, runs `uv build --wheel` against a disposable copy, requires exactly
one wheel, and inspects it as bounded, native-free `py3-none-any` bytes. A backend that emits local
credentials, VCS/environment/cache/editor/build state, native files, path injection, or an invalid
`RECORD` is refused. Creator separately syncs the locked dependencies into disposable storage with
`uv sync --locked --no-install-project`, then runs that environment's exact
`cozy-runtime --json --dir PROJECT describe`. The package project itself is not installed,
so Runtime derives the package descriptor without writing the source tree. The wheel's METADATA
carries the backend's declared requirements; `uv.lock` supplies the exact lock bytes. There is no
public profile, GPU, or custom-wheel selection.

Creator sends a canonical declaration containing only the source archive, lock, project
wheel, and package descriptor identities. Tensorhub returns presigned PUTs for missing objects;
Creator uploads them concurrently and finalizes with the byte-identical declaration.
Tensorhub derives compatible base worker profiles from package requirements and its own
image inventory. Exact replay is idempotent.

Until a real dependency requires more machinery, a package absent from approved base images
refuses instead of invoking a package index or native build. Model-bearing publication also
returns the typed `package_model_binding_deferred` refusal; binding intent stays in
`package.toml` for Tensorhub's future model-release resolver.

## Promotion

Publication does not move traffic. Tensorhub promotion is a separate explicit operator/policy
transaction; Creator has no promotion command.
