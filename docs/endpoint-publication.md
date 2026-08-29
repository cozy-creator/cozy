# Endpoint publication

Creator publishes endpoint code once and qualifies exact base profiles independently. It never
builds an endpoint image or accepts a Dockerfile/build command.

## Source release

The source directory must be a clean committed Git subtree containing:

- `pyproject.toml` with static project name/version and no build backend;
- `uv.lock` as opaque development provenance;
- `endpoint.toml` and `endpoint.descriptor.json`;
- pure Python project files; and
- `endpoint.release.json`; and
- optionally `endpoint.evaluated-config.json`.

`endpoint.release.json` always names a non-empty exact compatible-accelerator-model set. A
weightless endpoint uses explicit empty model arrays; a model-bearing endpoint declares complete
path-free facts:

```json
{
  "compatible_accelerator_models": ["NVIDIA H200"],
  "model_roots": [
    {"org": "cozy", "name": "my-model", "checkpoint_id": "sha256:<64 hex>"}
  ],
  "model_bindings": [
    {
      "path": "app.models.model",
      "checkpoint": {"org": "cozy", "name": "my-model", "checkpoint_id": "sha256:<64 hex>"},
      "config": {"assets": [], "document": {"digest": "sha256:<64 hex>", "length": 123}},
      "execution_layout": [
        {"component": "transformer", "root": {"org": "cozy", "name": "my-model", "checkpoint_id": "sha256:<64 hex>"}}
      ],
      "hardware_variant": "sm90"
    }
  ],
  "native_wheel_proof": {
    "fixture": "my_package.qualification:run",
    "expected_result_digest": "sha256:<64 hex>"
  }
}
```

Bindings sort by `path`; config assets sort by name; execution-layout order is construction order
and is preserved. Model roots and binding roots must be every-and-only equal.
`native_wheel_proof` is required exactly when measured custom-wheel contents include native bytes;
an oddly platform-tagged but actually pure wheel does not require it. The proof banks one
argument-free `module:callable` fixture and the canonical expected-result digest. Concrete device
selection remains a local/qualification-seat fact and never enters this portable declaration.

Publish with one or more explicit approved profiles—there is no default:

```sh
cozy endpoint publish org/endpoint \
  --release 1.0.0 \
  --profile torch2.13.0-cu126-cp312-linux-x86 \
  --profile torch2.13.0-cu130-cp312-linux-x86 \
  --reason "initial private release"
```

Creator builds a deterministic `py3-none-any` project wheel, canonicalizes descriptor/config
meaning, calls Tensorhub `begin`, uploads only requested roles through presigned PUTs, and calls
`finalize` with the byte-identical declaration. Exact replay converges without a client journal.
Finalize creates non-serving candidates; it spends no GPU money and moves no serving pointer.

## Custom wheels

Native dependencies are separate already-built exact wheels:

```sh
cozy endpoint publish org/endpoint --release 1.0.0 \
  --profile torch2.13.0-cu130-cp312-linux-x86 \
  --custom-wheel torch2.13.0-cu130-cp312-linux-x86=./dist/custom_op.whl \
  --reason "publish prebuilt custom op"
```

Creator verifies filename/METADATA/WHEEL/RECORD, tags, normalized distribution, import roots,
digest, and length. It never repairs, renames, retags, or builds a supplied wheel. Native source,
build files, `.pth` injection, base-distribution collisions, and native bytes inside the project
wheel refuse.

## Qualification and promotion

Hardware qualification and serving promotion are explicit Tensorhub operator/policy acts, not
Creator commands. Publication never rents a GPU or moves a serving pointer. Tensorhub qualifies
an exact candidate under a recorded cost bound, then atomically promotes explicit `vN/function`
targets after their evidence passes.

## Independently qualified local install

Local execution never borrows the paid OCI qualification. Tensorhub leases
`local-qualification-materials` only for a non-refused CPU/meta candidate with an exact registered
managed base; those bytes are inputs to Creator's independent local proof, not an execution grant:

```sh
cozy endpoint install org/endpoint@1.0.0 \
  --profile torch2.13.0-cu130-cp312-linux-x86 \
  --major v1 \
  --reason "qualify and install local release"
```

Managed bases live only at `$COZY_HOME/managed-bases/sha256/<realization hex>/`. The directory has
canonical `ManagedBaseReceipt.json`, exact `WheelhouseManifest.json`, and regular executable
`bin/python`, `bin/cozy-environment-proof`, `bin/cozy-runtime`, and—when native custom wheels are
present—`bin/cozy-native-wheel-proof`. The stored receipt digest must equal Tensorhub's managed
base-realization digest. Tensorhub never supplies a local path.

Creator downloads every-and-only the granted project/custom wheels, materializes without importing
endpoint code, measures the base host, runs the native operator proof when required, and only then
imports/checks the descriptor before atomically pinning. Control-install facts, portable Runtime receipt, managed-base
identity, current host evidence, and grant lease are stored separately. Missing/mutable base bytes
refuse; Creator performs no fresh dependency resolution, Torch/CUDA install, index access, or
native compilation.
