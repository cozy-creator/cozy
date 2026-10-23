# Release gate

Every first-party model runs for real on a fresh rental, and its numbers are held against the last passed gate on
the same SKU. Any FAIL blocks the publish. Design: tracker `design/ci-v2.md` §3.

## What a gate does

1. Rents one fresh machine of the SKU.
2. Puts the candidate on it: `--runtime-wheel`, `--tensorfs-wheel`, `--tensord` (a cozy binary), or `--packages`
   (a packages checkout whose directories run in place of the published packages).
3. Runs each cell once, with new prompts and seeds.
4. Ends the rental and checks that it is gone.
5. Runs `analyze.py`. A cell passes when it completed, served the last PASS's kernels (with no fallback), produced
   a sane output, and kept execution / step 1 / steady s/step / peak reserved VRAM within the last PASS's spread.

## Baselines

Assets of the `release-gate-baselines` release: `gate-<SKU>.json`, plus the reference image and clip the next gate's H3
cell starts from. Every gate reads them. A PASS (with `--publish`, or the workflow's `baseline` job) replaces them.
A SKU with no baseline needs `--establish` once.

## Running it

- **By hand:** `release-gate/gate.sh rtx-4090 --runtime-wheel <url#sha256=…> --publish`. It runs through the default
  `~/.cozy`, signed in as you.
- **From Actions:** `gh workflow run release-gate.yml -R cozy-creator/cozy-creator -f runtime_wheel=… -f changed='cozy-runtime v0.30.16'`.
  - Selection (`ci/affected`) picks the SKUs and cells; `skus` and `only` override it.
  - `release.yml` (cozy), cozy-runtime's `publish.yaml` and packages' `task publish` run it before they publish.

## CI sign-in

The runner signs in as the owner's account with its own device key: secret `COZY_CI_DEVICE_KEY` in the
`release-gate` environment of cozy-creator and cozy-runtime. To make or rotate the key, run
`release-gate/enroll-ci-key.sh <email>` in a terminal and type the emailed code. It runs
`cozy login <email> --export`, which leaves this computer's own key untouched.

## More cells and checks

- **Cells:** `--cells FILE` adds rows of `name<TAB>target<TAB>args`. The args are expanded in gate.sh, which defines
  `$SCENE`, `$CHAR`, `$R`, `$K`, `$M` and `$CLI`. A FILE ending in `.sh` is sourced after the last product cell.
- **Stress** (`cells/stress.sh`, suite `stress`): SDXL, Anima and 5 s H3 under 8, 6 and 3 GiB caps on one
  rtx-5090, with SDXL and Anima alternating three times at 6 GiB, plus SDXL and Anima on a real 8 GB rtx-3070.
  `stress.py` checks each cap took, nothing held more than its cap, the kernels match the uncapped cells, no bytes
  are left behind, and the alternation does not slow down. Dispatch with `-f stress=true` to force it. A new SKU
  (the rtx-3070) needs `-f skus=rtx-3070 -f establish=true` once.
- **Quality** (`quality.py`, every gate): CLIP prompt adherence of every render against its prompt (floor 0.25), and
  the gate's median CLIP-IQA sharpness against the last passed gates. Never a same-seed pixel diff.
- **Checks:** `analyze.py` runs every `check(gate) -> (fails, notes)` in `CHECKS` (`stress.check`, `quality.check`).
- **Known lists:** `known-blocking.json` and `known-failing.json` hold request-path blocking and lead-ruled skips;
  both may only shrink.
