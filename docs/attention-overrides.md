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
