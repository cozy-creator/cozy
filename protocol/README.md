# Native machine protocol

`cozy/machine/v1/` contains the generated bindings of the current machine API.
Its SOURCE file records the input source. Worker RPC bindings, wire-version metadata
and the old protocol corpus are retired. Historical record bytes have a small private
reader in internal/archive; they are not an active protocol or mutation API.
