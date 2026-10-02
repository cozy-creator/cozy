# Explicit owned machine endpoints

`cozy run <package/function> --machine-endpoint-file endpoint.json --await` selects
an existing owned machine without registering it as the controller's local machine
or as a rental. The public JSON record contains no credential:

```json
{
  "format": "cozy.machine.endpoint/1",
  "address": "host:port",
  "worker_id": "machine-id",
  "worker_boot_id": "current-boot-id",
  "tls_certificate_pem": "one pinned leaf certificate in PEM",
  "execution_workspace_id": "durable-workspace-id"
}
```

The existing controller signing key must already exist in the normal Cozy home,
and the machine must authorize its public key. The address selects a peer; the
pinned TLS leaf and per-operation signed Claim establish authority. The authenticated
workspace must match the record. Bad pins, identities, boots or workspaces stop
before request admission, with no local-machine or rental fallback. Additive JSON
fields and baseline protocol minor zero are accepted; operation capabilities decide
whether execution is available.

The CLI composes the existing business API, capture, orchestration and transport in
its own process. It writes normal install/request/output records in the existing
home, without taking the daemon lock, changing the machine roster, changing Hub
configuration, starting a local machine or performing daemon reconciliation. The
running daemon and global CLI installation need not change. An older controller
that cannot retain the selector is refused before any submission POST.

`cozy run watch <number-or-id>`, `show` and `cancel` use that request's retained
public selector. Closing the command or losing its watcher leaves machine work
durable; only explicit cancellation or an authored deadline cancels it. A selector
bound to an obsolete boot is refused rather than silently replaced. Refreshing a
recorded selector across boot/address changes is a remaining operation, not an
automatic target substitution.

Published packages and catalog models need the signed-in account's execution access
at the selected Hub. The CLI asks Tensorhub for access bound to the endpoint's pinned
leaf and the login device, and hands it to the machine with an owner-signed
`POST /v1/hubs/access`, exactly as for this computer's machine. The grant is cached
per endpoint (`machine/endpoints/<name>/`) and reused until the login, leaf or reset
generation changes or it nears expiry, so a run does not contact Tensorhub. Logout
erases every endpoint's grant and queues its removal for the next connection.

The explicit selector flag is not a claim that the chosen machine implements every
operation; jobs, browser selection and boot refresh remain separate gates.
