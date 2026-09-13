# Development rentals

`cozy rental new` accepts an explicit development mode and a caller-selected
SSH public-key file. The exact key and mode become part of the existing durable
paid request before the Hub is called.
No caller chooses an arbitrary OCI image or provider port.

To enable it for new manual and automatic rentals from this Creator home, set
`~/.cozy/config.yaml`:

```yaml
rentals:
  development: true
  ssh_public_key: ~/.ssh/id_ed25519.pub
```

Use an existing public key that you control. Relative config paths resolve beside
config.yaml; `~/` resolves to your home. The daemon reads configuration when it
starts, so restart it after changing defaults. Existing acquisition operations
replay their recorded mode and key, even if the defaults or key file change.
Existing running rentals keep their current mode.

For a single manual rental, use `cozy rental new <gpu> --development
--ssh-public-key <file>`. `cozy rental ssh-info <name>` returns its authenticated,
boot-pinned SSH address. Ordinary mode remains the default when development is
not enabled. Runtime error details remain available in ordinary mode too.

The Hub selects a registered development image for the normal GPU/CPU SKU and
returns its actual mapped SSH endpoint. The operator resolves that endpoint from
authenticated Hub readback when updating; Creator does not cache another SSH
address or introduce a SQLite migration.

The existing signed idle-development holder owns the daemon lock during updates.
The standard SSH/SFTP and Supervisord wrapper validates exact wheels, replaces the
quiescent Host/Runtime group and rolls back failure while preserving its Store.
Development mode does not choose an arbitrary image, expose a private key, or
change the normal direct worker request transport.
