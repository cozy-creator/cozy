# Request-scoped attention experiments

Ordinary requests use Runtime's environment and package attention policy. A
development or benchmark invocation can explicitly pin the backend for one
request:

```sh
cozy run paul/minimax-h3/fl2va prompt="A scene" \
  --attention-kernel=model/fl2va_dit=kitchen-int8
```

The value is `backend`, `component=backend`, or `model/component=backend`.
An unscoped backend applies to every attention site; prefer a component scope
when auxiliary encoders or VAEs need different attention capabilities. A bare
component selects that component name across model parameters; including the
model parameter makes the scope exact. For a turbo entrypoint the parameter is
`base_model`, for example `base_model/fl2va_dit=kitchen-int8`.

The existing positional spelling is equivalent:

```sh
cozy run paul/minimax-h3/fl2va prompt="A scene" \
  kernel.attention=model/fl2va_dit=kitchen-int8
```

Supply one override, without whitespace. Creator validates its syntax, transports
the exact string separately from the package payload, records it with the request
and includes it in idempotency/operation identity. Changing only the scope or
backend under an existing idempotency key refuses rather than replaying a
different computation. The invocation carries the existing `attention_kernel`
field; this syntax needs no new protocol field or database schema.

Runtime owns backend names, component matching, installed-kernel checks, hardware
and dtype support, numerical admission, and compiled/context-parallel restrictions.
An explicit pin never silently selects a different backend. Unqualified CP or
compiled combinations refuse; naming a kernel does not add support for them.
Scoped selection requires the corresponding Runtime implementation in the worker;
an older worker may transport the string but cannot interpret the new syntax.

This is a request-scoped development override, not a separate authorization mode
or a requirement to rent with `--development`. Existing local API credentials and
worker authorization remain unchanged. It neither changes a shared default nor
publishes a package policy. Package/model policy hooks are a separate Runtime
authoring facility. Overrides currently apply to serving callables.

## Optional prebuilt wheels on an existing rental

Use ordinary Python dependencies to add an optional attention implementation to
an unpublished package. The base worker image is the same in development and
ordinary modes. Development controls SSH access; it does not select a kernel
image, and SSH installation into the worker's global Python environment does not
install a dependency into the package's isolated environment.

Start with the package checkout you will benchmark and a compatible prebuilt
wheel. Keep the wheel inside that checkout so its bytes are captured with the
source. For example, after copying the wheel into `vendor/`:

```sh
uv add --no-sync ./vendor/cozy_kernel_flash_attn3-*.whl
cozy package install . --editable
cozy run local/my-package/generate --rental=my-rental --input=request.json \
  kernel.attention=model/dit=flash-attn3 --await --json
```

Replace the package, callable, model/component scope, and wheel filename with the
ones from your package. `uv add` writes the ordinary dependency and
`[tool.uv.sources]` wheel path and updates `uv.lock`. Creator captures the exact
wheel bytes and selected dependency closure; Runtime installs that closure in the
package environment on the named rental. `--rental` selects an existing machine
and never buys a replacement. This path requires Linux amd64 and a CPython version in the Runtime supported window,
plus wheels compatible with the package's Torch/CUDA dependencies and the target
GPU. Do not infer hardware compatibility from successful local installation.

A selected library extra can include the kernel wheel instead, using standard
`my-library[kernel-extra]` dependency syntax. An unselected optional group does not
install anything. Adding a wheel and selecting an attention backend are separate
steps: a request override never downloads a missing kernel. Runtime validates the
selected implementation and refuses unsupported combinations.

For an A/B/A comparison, keep the request input and random seed fixed, run each
backend with a fresh request identity, and record Runtime's selected backend and
artifact provenance along with the outputs and elapsed time. When replacing a
wheel, retain each original wheel and rerun `uv add` and the editable install to
capture the new bytes. Never replace a captured wheel in place under `~/.cozy`.
The same package version can have different captured source/dependency identities;
reusing an idempotency key must never turn one computation into another.
