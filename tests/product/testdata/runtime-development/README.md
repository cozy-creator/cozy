# Temporary coordinated Runtime fixture

Creator wire55 needs a wire55 Runtime. Installing published0.16.10 (wire49) made
full CI repeatedly wait for a daemon that correctly refused startup. A public
Runtime release is not required for every coordinated development change.

`peer.json` identifies the exact pure-Python wheel and source revision used by the
cohort. CI verifies the wheel SHA-256 and embedded COMMIT before installation,
then checks the real version verb against both that source and the required wire
minor before starting product tests. Private child fixtures receive the same
wheel through the existing `-child-runtime-wheel` option. No credentials or GPU
kernels are embedded in this test-only wheel.

The wheel was built from an immutable Git archive of Runtime
`f084c910f0ce5bd5027905d592c0099621009fac`, stamped by that revision's existing
`scripts/stamp-build-provenance.py`, and built with `uv build --wheel`. The source
archive, build output, SHA readback, host-tool qualification and exact worker-image
proof are retained under `outputs/h3-general-lora-20260914/runtime` in the owning
workspace. PyPI, Git tags and production processes were not changed to create it.

To replace it, build the next reviewed immutable Runtime revision the same way,
place the wheel beneath its SHA directory, update peer.json, and run the full
product suite. Remove this fixture and return CI to the qualified normal release
when a real Runtime release covers this cohort. Do not update to a branch name,
restamp a live dirty source tree, or relax compatibility to make a stale tool pass.
