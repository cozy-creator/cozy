# Private rental software updates

`cozy rental update NAME` changes an existing private rental's Runtime/TensorFS pair. It is an
operation like a package install: a numbered run of kind `update` (`cozy run
list/show/watch/cancel`), sent to the rental's machine as Run kind: update.

- **What it installs:** `--runtime-version`/`--tensorfs-version` published releases, or
  `--runtime-wheel` (optionally with `--tensorfs-wheel`) local wheels, measured where they are
  (SHA-256, length) and written to the machine as measured. With nothing named, the Hub's target
  pair, named exactly. A pair the machine already runs is answered unchanged.
- **On the machine (TensorD):** stages and verifies the pair, closes admission, waits for the work
  it admitted, restarts on the pair in place (same boot) and commits once the new service proves
  ready, or rolls back. A published wheel download that stops moving fails on its measured
  stillness.
- **Cancel:** `cozy run cancel N` ends it before activation (`canceled`); once activating it ends
  on its own (`cancel_unavailable`).
- **The rental's other work** is not sent while the update is unsettled, so it lands on the new
  software. Reads and cancels of other runs go through.
- **A new boot** takes its Hub's target the same way, as an update run: pinned, exactly; unpinned,
  only forward from an older image. `rental new` shows the result when the Hub pins.

An interrupted CLI does not cancel the run; a restarted Calcifer attaches to the same run.
