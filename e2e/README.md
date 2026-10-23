# e2e

Real runs of `cozy` on GitHub's runners, chosen by what a change touches (CI v2: `tracker/design/ci-v2.md`).

- `ci/affected <repo> <base> <head>` prints what a change must run; `ci/suites.toml` holds the rules.
- `cli/run.sh`: install, up, the local machine, the change's candidates, BiRefNet on the CPU, a second executor, down, uninstall.
- `mps/run.sh`: BiRefNet and SDXL on a Mac's GPU (explicit `select` only, until it can pass).
- `.github/workflows/e2e.yaml` runs the selected suites on PRs here and for cozy-runtime, packages and the release gate.

Run one suite on a fresh computer: `e2e/cli/run.sh --out /tmp/e2e [--cozy-bin ./cozy] [--runtime-wheel …] [--tensorfs-wheel …]`.
