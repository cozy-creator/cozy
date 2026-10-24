# Release gate

Release testing is manual: the release agent picks the suites a release needs (`--suite`, no default), runs them, and
promotion needs the passing run (`verify.sh`). Every cell is a real run held against the last passed gate on the same
SKU; any FAIL blocks the promotion. Design: tracker `design/ci-v2.md` §3.

## Suites

| Suite | Proves | Rentals (`skus`) | Rough cost · time | Pick it for |
|---|---|---|---|---|
| `runtime` | every first-party package's default call (H3 15 s with a described reference, SeedVR2, SDXL, Klein, Qwen, Anima, BiRefNet, the Qwen3.5 describer, Fibo, HiDream, MOSS, Qwen-Image + its one-release swap), then SDXL in a second executor: kernels, times, VRAM, quality, LM coverage | rtx-4090, rtx-5090 | ~$6 · ~75 min (in parallel) | a Runtime, TensorFS, varena or TensorD change; any rewrite |
| `packages` | the packages `--pins` (or `--only`) names, plus the SDXL smoke | rtx-4090, rtx-5090 | ~$1–3 · 15–30 min | a package bump |
| `tensord` | the machine agent through `cozy`: generations installed, updated and rolled back; run and executor lifecycle; GPU leases across co-resident models; multi-GPU placement; no re-execution; output delivery; cl-650's update and floor checks | one cheap 2-GPU pod | ~$1 · ~45 min | a TensorD or cozy change (building) |
| `varena` | memory under hard limits (STRESS agent, building) | see its `skus` | | a memory-manager change |
| `tensorfs`, `hub` | TensorFS and the Hub (TFS-HUB agent, building) | see their `skus` | | a TensorFS or Hub change |
| `cli` | the CLI as users drive it (COVERAGE agent, building) | see its `skus` | | a CLI or Calcifer change |

A rewrite picks every suite. Suites stack: `--suite runtime,packages` runs each cell once.

A suite is `suites/<name>/`:
- `cells.sh`, sourced in gate.sh: `cell`, `pkg`, `product`, `expect`, `log`, `$CLI`, `$M`, `$R`, `$REF` and the scenario;
- optional `before.sh`, sourced on the fresh pod before the candidate goes on;
- optional `check.py` with `check(gate) -> (fails, notes)`, which analyze.py runs for that suite;
- `skus`, a `<sku> [gpus]` per line, which the workflow rents.

`$OUT/suites` and the verdict's `SUITES` line name what ran. A known-failing.json entry's `"cells"` (exact names) parks
those cells on its SKU; deleting the entry un-parks them.

## What a gate does

1. Rents one fresh machine of the SKU.
2. Puts the candidate on it: `--runtime`, `--tensorfs`, `--tensord` (a cozy binary; each a file or `URL#sha256=`), `--pins`
   (exact package releases, `sdxl@2.13.0,…`: a package candidate is published and yanked, and installs only by its pin),
   or `--packages` (a packages checkout whose directories run). Unpinned cells run the root package users get.
3. Runs the chosen suites' cells, each once, with new prompts and seeds.
4. Ends the rental and checks that it is gone.
5. Runs `analyze.py`. A cell passes when it completed, served the last PASS's kernels (with no fallback), produced
   a sane output, and kept execution / step 1 / steady s/step / peak reserved VRAM within the last PASS's spread.

## Baselines

Assets of the `release-gate-baselines` release: `gate-<SKU>.json`, plus the reference image and clip the next gate's H3
cell starts from. Every gate reads them. A PASS (with `--publish`, or the workflow's `baseline` job) replaces them.
A SKU with no baseline needs `--establish` once.

## Running it

- **By hand:** `release-gate/gate.sh rtx-4090 --suite runtime --runtime <url#sha256=…> --publish`. It runs through the
  default `~/.cozy`, signed in as you.
- **From Actions:** `gh workflow run release-gate.yml -R cozy-creator/cozy-creator -f suites=runtime,packages -f pins=… -f runtime_wheel=…`.
  The suites' `skus` pick the rentals; `skus` and `only` override them. `release.yml -f suites=…` gates a cozy candidate.

## Promotion

Nothing reaches users before a PASS. Candidates wait where users don't look: cozy as a GitHub prerelease (`cozy update`,
the installer and Homebrew follow latest), TensorFS and the Runtime as GitHub prerelease assets (not on PyPI), and
packages as yanked releases. `candidate.json` in each gate's output names the exact set. `verify.sh <run|dir>...` prints it
only when every SKU passed. `promote.sh <run|dir>...` then hands that set to users in one step: the same files go to
PyPI, the cozy prerelease becomes latest plus Homebrew, and the packages are un-yanked (`cozy package yank --undo`).
The worker image and the Hub pin follow, by the release owner.

## CI sign-in

The runner signs in as the owner's account with its own device key: secret `COZY_CI_DEVICE_KEY` in the
`release-gate` environment of cozy-creator and cozy-runtime. To make or rotate the key, run
`release-gate/enroll-ci-key.sh <email>` in a terminal and type the emailed code. It runs
`cozy login <email> --export`, which leaves this computer's own key untouched.

## More cells and checks

- **Cells:** `--cells FILE` adds rows of `name<TAB>target<TAB>args`. The args are expanded in gate.sh, which defines
  `$SCENE`, `$CHAR`, `$R`, `$K`, `$M` and `$CLI`. A FILE ending in `.sh` is sourced after the last product cell.
- **CLI behaviour** (`--before cells/cl650.sh`, sourced on the fresh pod before the candidate goes on): cozy #1925
  updates a real 0.11.15 TensorD and verifies the version, and a release whose cozy-runtime floor the pair misses
  refuses `runtime_floor_unmet` until `cozy machine update --runtime <v> --tensorfs <v>`. Each check is a gate.sh
  `expect` row in `expect.tsv`, which `analyze.py` fails on.
- **One release, two checkpoints** (`swap.py`): Qwen-Image text → edit → text and its two Lightning LoRAs in turn, each group on one executor with no restart (run 15744: TFS 0.15.0's `{release}/{component}` set names collided). No first-party SDXL LoRA exists to swap.
- **LM coverage** (`lm.py`, every gate): Fibo's and Klein's prompt upsamplers and H3's reference describer ran their
  greedy token loops (H3's cell gives a character reference, which BiRefNet cleans and Qwen3.5 describes).
- **Stress** (`cells/stress.sh`, suite `stress`): SDXL, Anima and 5 s H3 under 8, 6 and 3 GiB caps on one
  rtx-5090, with SDXL and Anima alternating three times at 6 GiB, plus SDXL and Anima on a real 8 GB rtx-3070.
  `stress.py` checks each cap took, nothing held more than its cap, the kernels match the uncapped cells, no bytes
  are left behind, and the alternation does not slow down. Dispatch with `-f stress=true` to force it. A new SKU
  (the rtx-3070) needs `-f skus=rtx-3070 -f establish=true` once.
- **Host RAM** (`cells/stress-host.sh`, suite `stress-host`): Klein and Qwen on a CPU rental under its cgroup
  `memory.max`, which `cells/host-cap.py` measures; `stress.py` wants the cap below the RAM and page/spill bytes in
  each record. The CLI caps no host memory, so the rental's container is the cap. Parked (known-failing, xs-058).
- **Quality** (`quality.py`, every gate): CLIP prompt adherence of every render against its prompt (floor 0.25), and
  the gate's median CLIP-IQA sharpness against the last passed gates. Never a same-seed pixel diff.
- **Checks:** `analyze.py` runs every `check(gate) -> (fails, notes)` in `CHECKS` (`stress.check`, `quality.check`).
- **Known lists:** `known-blocking.json` and `known-failing.json` hold request-path blocking and lead-ruled skips;
  both may only shrink.
