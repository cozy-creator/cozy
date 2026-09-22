# Daemon shutdown and retained execution

`cozy down` stops the Creator client daemon without canceling requests or ending
rentals. Retained paused, blocked, and completed work does not require the client
to stay online. Its records, outputs, and rental holds remain available after
`cozy up`.

A durable Runtime acceptance receipt permits both detached local and remote
execution to continue independently. Reconnection reads the same execution and
workspace; it does not submit another execution or cancel the existing one.
Preparation without acceptance, unfinished input transfer, pending execution
control, and active work owned by a legacy local daemon child can still require
the daemon. Normal shutdown names those dependencies and refuses.

`cozy down --force` overrides that dependency guard and disconnects the client.
It does not send request cancellation or rental deletion. Legacy local execution
owned by the daemon may be interrupted and uses the existing recovery rules on
restart; force does not promise that every process continues.

`cozy down --all` retains its separate destructive meaning: cancel work and end
rentals. Automatic idle shutdown continues to use the full retention and resource
obligations rather than the explicit disconnect policy.
