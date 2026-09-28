# Private rental maintenance

New Creator private rentals use maintenance-capable worker images by default.
Creator creates a private OpenSSH identity under `~/.cozy/auth/` unless an explicit
operator public key is configured. Only the public key enters the rental request.
The exact image mode and key remain part of the durable acquisition; changing
local defaults never changes an existing rental or a replayed acquisition.

```sh
cozy rental new h100-sxm5-80gb
cozy rental update <name>
```

`cozy rental update` selects the approved SDK pair from the active compatible Hub
image profile. It updates only Runtime/TensorFS, retaining model bytes, intermediate
work and the heavy Torch/CUDA environment. Active execution is never interrupted.
The live Creator daemon holds only the target rental out of new dispatch, and the
existing worker guardian owns installation, rollback and status across disconnects.
Another rental and the local daemon remain available.

For an operator-owned SSH identity, set `rentals.ssh_public_key` to its public key
path or pass `--ssh-public-key FILE` on acquisition. Relative config paths resolve
beside config.yaml; `~/` resolves to your home. The corresponding private key must
remain available for updates. `cozy rental ssh-info NAME` reads the current pinned
SSH endpoint. Configuration changes take effect on the next daemon start.

Use `--development=false` or `rentals.development: false` to explicitly select an
immutable private worker. Shared/public worker image policy is separate. Existing
images without guarded replacement and durable update support require a current
maintenance-capable image; the update command never replaces or buys a rental.
