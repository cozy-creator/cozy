# Explicit development rentals

`cozy rental new` will accept an explicit development mode and a caller-selected
SSH public-key file. The exact key and mode become part of the existing durable
paid request before the Hub is called. Normal invocations omit development intent.
No caller chooses an arbitrary OCI image or provider port.

The Hub selects a registered development image for the normal GPU/CPU SKU and
returns its actual mapped SSH endpoint. The operator resolves that endpoint from
authenticated Hub readback when updating; Creator does not cache another SSH
address or introduce a SQLite migration.

The existing signed idle-development holder owns the daemon lock during updates.
The standard SSH/SFTP and Supervisord wrapper validates exact wheels, replaces the
quiescent Host/Runtime group and rolls back failure while preserving its Store.
This change neither activates development workers by default nor launches a paid
resource during implementation validation.
