# Calcifer

Calcifer is Cozy's personal controller: the local API, web UI, durable submission outbox,
machine observation, history and output collection. `cozy` remains the public CLI.
`cozy up` and commands that need a controller launch a private `calcifer` process; where
systemd user units are available, it runs as `calcifer` or `calcifer-<home hash>`.

The rename preserves the current home, `creator.sqlite`, machine/account identities,
`daemon.lock`, `daemon.log`, `daemon.idle_shutdown_s`, HTTP routes and JSON fields.
`cozy daemon log` continues to read the same log. Plain `cozy down` detaches the controller
and leaves durable machine work running; `cozy down --all` remains explicit cancellation.

Installing a new CLI does not replace a live controller. Commands follow its existing
lock and API address. At the owner's cutover, `cozy down` followed by `cozy up` starts
Calcifer against the retained home and history. No automatic shutdown or state migration
is part of the rename.
