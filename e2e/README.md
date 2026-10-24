# e2e

Real runs of `cozy` on GitHub's runners, chosen by what a change touches (CI v2: `tracker/design/ci-v2.md`).

- `ci/affected <repo> <base> <head>` prints what a change must run; `ci/suites.toml` holds the rules.
- **The candidate set** (`candidates/`): on Linux, TensorD, TensorFS and the Runtime, built from the change at its commit and the
  other repos at master (`stack_ref`, `runtime_ref`), go on the local machine together, so a mismatch between them fails the PR.
- `cli/run.sh`: install, up, the local machine, the candidates, a weightless request through them, BiRefNet on the CPU, a second
  executor, down, uninstall.
- `imports/run.sh`: every published first-party package release installs (`cozy package install`) and imports on the candidates;
  a Runtime that moves what a published package imports fails here.
- `mps/run.sh`: BiRefNet and SDXL on a Mac's GPU (explicit `select` only, until it can pass).
- `.github/workflows/e2e.yaml` runs the selected suites on PRs here and for cozy-runtime, packages and the release gate.

Run one suite on a fresh computer: `e2e/cli/run.sh --out /tmp/e2e [--cozy-bin ./cozy] [--runtime-wheel …] [--tensorfs-wheel …]`.
Against chosen versions: `gh workflow run e2e.yaml -f repo=cozy-runtime -f ref=<sha> -f stack_ref=<cozy-creator sha> -f select="cli imports build:runtime"`.

## Merging (Paul, 2026-10-24)

CI runs only in the merge queue, on the final PR + branch state, for PRs into `master` and `rc/*`. A pull request
runs placeholders, so it can be queued; nothing runs on a push. Merge with `gh pr merge --squash`, which enqueues,
never `--admin`. A direct push to master (admins) skips CI. Required checks:

- cozy-creator: `build`, `e2e`
- cozy-runtime, packages: `build`, `e2e / e2e`
- tensorhub: `build`, required once Actions runs there again (the org's budget)
