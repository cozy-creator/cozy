# Coordinated development Runtime in Creator CI

The product suite builds the exact Runtime source commit named by
`COZY_RUNTIME_SOURCE` in `.github/workflows/ci.yaml`. The existing read-only
`COZY_RUNTIME_READ_KEY` checks out source; checkout does not persist credentials.
`scripts/build-runtime-peer.py` archives that commit and records source archive,
recipe and wheel digests. The CPU-only test environment and isolated test shards
use that same wheel, with no SDK release or PyPI publication.

Creator's interface generator ABI must match the captured Host Runtime (currently
`cozy.interface-generator/6` for authored Model adapters). A mismatch receives a
specific incompatibility error before accepting any generated wheel. The generated
protocol, Runtime source, TFS dependency floor and ordinary native/default/adapter
checks qualify the cohort together. Updating a version string alone is not proof.
