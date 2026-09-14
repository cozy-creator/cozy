# Coordinated development Runtime in Creator CI

The product suite builds the immutable Runtime revision named by
`COZY_TEST_RUNTIME_REV` in `.github/workflows/ci.yaml`. It uses the existing
`COZY_RUNTIME_READ_KEY`, the read-only Runtime deploy key owned by Creator CI;
checkout removes the credential before subsequent build/test commands.

Builds use an exact Git archive, Runtime's own `stamp-build-provenance.py`, and
`uv build --wheel`. CI records the resulting wheel SHA, verifies the installed
source commit and required wire minor before starting any daemon, then supplies
that same wheel to private child fixtures. The roster job uses the same existing
read-only checkout mechanism instead of an absent token and a silent unarmed leg.

This is a test dependency, not a PyPI release or a shipped Creator dependency.
For the next coordinated update, change the pinned immutable Runtime commit and
qualify its wheel and product suite together. A normal release may replace the
source build when the owner chooses to release; a bugfix does not require one.
