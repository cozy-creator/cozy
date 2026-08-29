# Endpoint publication

Creator publishes endpoint code. It never builds an endpoint Docker image or selects a GPU.
It invokes the project's declared PEP 517 backend locally through `uv build --wheel`.

## Source release

The source directory is the current working tree and contains:

- static project metadata and dependencies in `pyproject.toml`;
- an exact `uv.lock`;
- author intent in `endpoint.toml`;
- pure Python project files.

No Git repository, commit, or clean-tree state is required. Modified and ordinary untracked files
participate normally. Creator omits `.env*`, key/credential material and directories, VCS metadata,
virtual environments, caches, editor state, and build output from the provenance archive. If a
build backend puts any such local material into the wheel, wheel inspection refuses it.

`endpoint.toml` is the only author configuration. `endpoint.release.json` and
`endpoint.evaluated-config.json` are retired, as is committed `endpoint.descriptor.json`.
The final project wheel refuses native files, nested wheels, credentials, symlinks, unsafe paths,
and every tag except `py3-none-any`. Build configuration may exist in source; only the produced
wheel crosses the execution boundary.

```sh
cozy endpoint publish org/endpoint \
  --release 1.0.0 \
  --reason "initial release"
```

Creator syncs the locked dependencies into disposable storage with
`uv sync --locked --no-install-project`, then runs that environment's exact
`cozy-runtime --json --dir PROJECT describe`. The endpoint project itself is not installed,
so Runtime derives the descriptor without writing the source tree. Creator then runs
`uv build --wheel` in private output staging. `uv` is the frontend; the backend declared in
`[build-system]` chooses the package files and writes the wheel. Creator does not rewrite that
result: it verifies the ZIP/RECORD, metadata identity, import roots, secret/native exclusions, and
exact `py3-none-any` tag, then hashes the wheel bytes. `uv.lock` supplies the exact lock bytes.
There is no public profile, GPU, or custom-wheel selection.

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
