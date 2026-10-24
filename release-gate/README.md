# Release gate

Release testing is manual: the release agent picks the suites a release needs (`--suite`, no default), runs them, and
the promotion names the passing run (`verify.sh`). Every cell is a real run held against the last passed gate on the
same SKU and suites; any FAIL blocks the promotion. Design: tracker `design/ci-v2.md` §3.

## Suites

| Suite | Proves | Rentals (`skus`) | Cost · time | Pick it for |
|---|---|---|---|---|
| `runtime` | every first-party package's default call (H3 15 s with a described reference, SeedVR2, SDXL, Klein, Qwen, Anima, BiRefNet, the Qwen3.5 describer, Fibo, HiDream, MOSS, Qwen-Image and its one-release swap), then SDXL in a second executor: kernels, times, VRAM, quality, LM coverage | rtx-4090, rtx-5090 | ~$6 · ~75 min, in parallel | a Runtime, TensorFS, varena or TensorD change; any rewrite |
| `packages` | the packages `--pins` (or `--only`) names, plus the SDXL smoke | rtx-4090, rtx-5090 | ~$1–3 · 15–30 min | a package bump |
| `varena` | the memory manager: cold load, co-residency, reload, 8/6/3 GiB caps, a real 8 GB card, host RAM (below) | its `skus` | | a varena or memory change |
| `tensord` | the machine agent through `cozy`: generations, run and executor lifecycle, leases, multi-GPU, no re-execution, output delivery, cl-650 (building) | a cheap 2-GPU pod | | a TensorD or cozy change |
| `tensorfs` | downloads as fast as the last pass (Hub and datacenter cache), nothing from the Hub again, resume after a cut; uploads of new weights, dedup of a re-upload, resume after a cut; a straggler re-asked, not waited out (below) | cpu | ~$0.3 · ~40 min | a TensorFS, download or upload change |
| `hub` | the API contract (`api/openapi.yaml`): anonymous paths, then account, rental, package publish/yank/pin/un-yank and model upload/publish/yank/delete through `cozy` (below); the migration rehearsal before a deploy | cpu | ~$0.2 · ~30 min | any Hub release |
| `cli` | (building) | | | a CLI change |

A rewrite picks every suite. Suites stack on one rental per SKU (`--suite runtime,packages` runs each cell once,
against `gate-<SKU>-runtime-packages.json`). A suite is `suites/<name>/`: `cells.sh` (sourced in gate.sh: `cell`,
`pkg`, `product`, `expect`, `log`, `$CLI`, `$M`, `$R`, `$REF` and the scenario), optional `before.sh` (the fresh pod,
before the candidate goes on), optional `check.py` (`check(gate) -> (fails, notes)`, listed in analyze.py), and `skus`
(`<sku> [disk GB] [gpus]` per line, the workflow's rentals). The verdict's `SUITES` line names what ran.

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

Assets of the `release-gate-baselines` release: `gate-<SKU>-<suites>.json`, plus the reference image and clip the next gate's H3
cell starts from. Every gate reads them. A PASS (with `--publish`, or the workflow's `baseline` job) replaces them.
A SKU and suites with no baseline need `--establish` once.

## Running it

- **By hand:** `release-gate/gate.sh rtx-4090 --suite runtime --runtime <url#sha256=…> --publish`, through the
  default `~/.cozy`, signed in as you.
- **From Actions:** `gh workflow run release-gate.yml -R cozy-creator/cozy-creator -f suites=runtime,packages -f pins=…`.
  The suites' `skus` pick the rentals; `skus` overrides them, `only` narrows the cells. `release.yml -f suites=…`
  gates a cozy candidate before it is promoted.

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
- **varena** (`suites/varena`): the memory manager through its paces on SDXL, Klein and 5 s H3 on Ampere, Ada,
  Blackwell and a CPU rental (Hopper/B200 on request): cold load, co-residency, unload and reload, 8/6/3 GiB caps with
  SDXL and Klein alternating at 6, a real 8 GB card, and host RAM under the rental's cgroup with page/spill to disk.
  `check.py` holds each load and refill to a share of the machine's measured ceiling (`ceiling.py`), checks caps,
  alternation and host RAM; `analyze.py`, time and peak VRAM. Selected by `varena` (varena, TensorFS, TensorD, the
  Runtime); a SKU new to it needs `-f suites=varena -f skus=<sku> -f establish=true` once.
- **tensorfs** (`suites/tensorfs`, one `cpu` rental, ~$0.3, ~40 min, `-f suites=tensorfs`): pick it for a TensorFS or
  download/upload path change. It shows TensorFS moves weights as fast as before and dedups:
  - `probe` (on the rental): the machine's disk, written and read cold (the varena method); an executor never dials
    (cr-012), so there is no network ceiling, and downloads are held against the last pass.
  - `model download` of a large many-object model cold (Hub and datacenter-cache times are separate baselines), again
    (nothing from the Hub), and cut part-way then resumed (only the rest comes).
  - `model upload` of new random weights from the gate's host to a throwaway `<account>/ci-tfs-<R>` (the time
    includes staging from the host: run it from a runner), the same folder to a second tag (dedup: the same tree, a
    tenth of the time), and new weights cut part-way then resumed. The model is deleted after.
  - `straggler` (on the gate's host, where a client may dial): `tfs download` of BiRefNet clean and through
    `straggler.py`, which drips its first objects as a slow storage GET does. A straggler must be re-asked, not
    waited out (at most 1.5x clean + 20 s).
- **hub** (`suites/hub`, one `cpu` rental, ~$0.2, ~30 min, `-f suites=hub`): pick it for any Hub release. It shows the
  API contract (tensorhub `api/openapi.yaml` at `$HUB_REF`, read with `gh`) holds as cozy uses it:
  - `contract.py`: each path no verb reaches, called anonymously, answers as its contract says (200 with a body its
    schema admits, or 401 when bearer-only). This is the one place in release-gate that speaks HTTP to the Hub by
    hand: it is the contract test.
  - Through `cozy`, on a throwaway `<account>/ci-hub-<R>`: whoami, credits, credits history, the rental catalog and
    the rental itself; a package locked, published twice, run, yanked (the simple index marks it `data-yanked`),
    resolved past, run by exact pin, and un-yanked; a model uploaded, published (listed), downloaded, yanked (listed
    yanked), downloaded by pin, un-yanked, deleted (gone). The package's releases are yanked and the model deleted.
  - Before a Hub deploy, the release agent also rehearses its migrations: dump prod (in-pod `pg_dump -Fc`, copied
    in sha256-checked chunks, a full `pg_restore -f /dev/null`), restore it into a scratch Postgres, start the
    candidate image against it (it migrates on start) and wait for `/healthz` 204. Record the dump and the result
    with the gate.
- **Quality** (`quality.py`, every gate): CLIP prompt adherence of every render against its prompt (floor 0.25), and
  the gate's median CLIP-IQA sharpness against the last passed gates. Never a same-seed pixel diff.
- **Checks:** `analyze.py` runs every `check(gate) -> (fails, notes)` in `CHECKS` (quality, lm, swap, and a suite's).
- **Known lists:** `known-blocking.json` and `known-failing.json` hold request-path blocking and lead-ruled skips;
  both may only shrink. A known-failing entry's `cells` names the suite cells it parks (`sku` "*": every SKU).
