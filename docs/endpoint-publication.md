# Endpoint publication

Creator publishes endpoint code. It never builds an endpoint Docker image, selects a GPU,
or runs a project build backend.

## Source release

The source directory must be a clean committed Git subtree containing:

- static project metadata and dependencies in `pyproject.toml`;
- an exact `uv.lock`;
- author intent in `endpoint.toml`;
- pure Python project files.

`endpoint.toml` is the only author configuration. `endpoint.release.json` and
`endpoint.evaluated-config.json` are retired, as is committed `endpoint.descriptor.json`.
Native files, build recipes, nested wheels, model weights, credentials, symlinks, and
unsafe paths refuse.

```sh
cozy endpoint publish org/endpoint \
  --release 1.0.0 \
  --reason "initial release"
```

Creator syncs the locked dependencies into disposable storage with
`uv sync --locked --no-install-project`, then runs that environment's exact
`cozy-runtime --json --dir PROJECT describe`. The endpoint project itself is not installed,
so Runtime derives the descriptor without writing the source tree. Creator then
deterministically builds one `py3-none-any` project wheel. Its METADATA carries
`Requires-Python` and `Requires-Dist` directly from `pyproject.toml`; `uv.lock` supplies the
exact lock bytes. There is no public profile, GPU, or custom-wheel selection.

Creator sends a canonical declaration containing only the source archive, lock, project
wheel, and descriptor identities. Tensorhub returns presigned PUTs for missing objects;
Creator uploads them concurrently and finalizes with the byte-identical declaration.
Tensorhub derives compatible base worker profiles from package requirements and its own
image inventory. Exact replay is idempotent.

Until a real dependency requires more machinery, a package absent from approved base images
refuses instead of invoking a package index or native build. Model-bearing publication also
returns the typed `endpoint_model_binding_deferred` refusal; binding intent stays in
`endpoint.toml` for Tensorhub's future model-release resolver.

## Promotion

Publication does not move traffic. Tensorhub promotion is a separate explicit operator/policy
transaction; Creator has no promotion command.
