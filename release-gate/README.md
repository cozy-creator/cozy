# Release gate

Every first-party model runs for real on a fresh rental, and its numbers are held against the last passed gate on
the same SKU. Any FAIL blocks the publish. Design: tracker `design/ci-v2.md` §3.

## What a gate does

1. Rents one fresh machine of the SKU.
2. Puts the candidate on it: `--runtime-wheel`, `--tensorfs-wheel`, `--tensord` (a cozy binary), `--pins`
   (exact package releases, `sdxl@2.13.0,…`: a package candidate is published and yanked, and installs only by its pin),
   or `--packages` (a packages checkout whose directories run). Unpinned cells run the root package users get.
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
  - Testing is manual (Paul 10-24): only the release agent dispatches it, with the suites it chose. No workflow calls it.

## Promotion

Nothing reaches users before a recorded PASS. Candidates wait where users don't look: cozy as a GitHub prerelease
(cozy-publish.yaml; `cozy update`, the installer and Homebrew follow latest), TensorFS and the Runtime as GitHub
prerelease assets (tensorfs-publish.yaml `candidate=true`, cozy-runtime runtime-publish.yaml `mode=source`), and packages
as yanked releases (`task publish`). The release agent picks the suites, runs the gate by hand or from Actions, and
records the result. `candidate.json` in each gate's output names the exact set; `verify.sh <run|dir>...` prints it only
when every SKU passed. `promote.sh <run id>` (or `--record <URL> <dirs>...` for a hand gate) hands that set to users in
one step: the same files to PyPI, the cozy prerelease latest plus Homebrew (cozy-promote.yaml), the packages un-yanked
(`cozy package yank --undo`). The worker image and the Hub pin follow, by the release owner.

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
