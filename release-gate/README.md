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
  - `release.yml` (cozy), cozy-runtime's `publish.yaml` and packages' `task publish` run it before they publish.

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
- **Suites** (`--suite <name>`): a suite runs instead of the product cells, from `suites/<name>/cells.sh`, judged by its
  `check.py`, against its own baseline `gate-<SKU>-<name>.json`. Its `skus` lists the SKUs the workflow rents for it.
- **varena** (`suites/varena`): the memory manager through its paces on SDXL, Klein and 5 s H3 on Ampere, Ada,
  Blackwell and a CPU rental (Hopper/B200 on request): cold load, co-residency, unload and reload, 8/6/3 GiB caps with
  SDXL and Klein alternating at 6, a real 8 GB card, and host RAM under the rental's cgroup with page/spill to disk.
  `check.py` holds each load and refill to a share of the machine's measured ceiling (`ceiling.py`), checks caps,
  alternation and host RAM; `analyze.py`, time and peak VRAM. Selected by `varena` (varena, TensorFS, TensorD, the
  Runtime) or `-f varena=true`; a SKU new to it needs `-f skus=<sku> -f varena=true -f establish=true` once.
- **Quality** (`quality.py`, every gate): CLIP prompt adherence of every render against its prompt (floor 0.25), and
  the gate's median CLIP-IQA sharpness against the last passed gates. Never a same-seed pixel diff.
- **Checks:** `analyze.py` runs every `check(gate) -> (fails, notes)` in `CHECKS` (quality, lm, swap, and a suite's).
- **Known lists:** `known-blocking.json` and `known-failing.json` hold request-path blocking and lead-ruled skips;
  both may only shrink. A known-failing entry's `cells` names the suite cells it parks (`sku` "*": every SKU).
