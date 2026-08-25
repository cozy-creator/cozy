# Exit matrix (frozen)

Copied verbatim from tracker-v2 `cozy-runtime-cli.md`. `cozy` uses it unchanged — no
local-only codes. `scripts/fence.py` checks `internal/exit/exit.go` against this table
(code, name and meaning, row for row); editing one side alone turns the fence red.

| code | name | meaning |
|---|---|---|
| 0 | ok | success, including idempotent no-ops |
| 1 | internal | unexpected fault — a bug, never a user condition |
| 2 | usage | bad invocation: unknown flag, malformed `key=value`, majorless target |
| 3 | validation | typed payload/schema/bounds refusal; verifier refusal at ingest |
| 4 | not_found | unknown ref/function/endpoint/attempt; hub 404 rendered verbatim |
| 5 | credential | private/gated source without a credential — names the credential to add |
| 6 | structural | structural incompatibility — never a fit shortfall (a degradable shortfall degrades and exits 0; a below-floor shortfall is 14) |
| 7 | confirm | destructive op without `--yes` |
| 8 | offline_miss | `--offline` and the bytes are not in the CAS |
| 9 | unavailable | socket / local server / hub unreachable |
| 10 | deadline | `--timeout` or request deadline exceeded |
| 11 | failed | attempt/job failure terminal (remedy verbatim) |
| 12 | canceled | canceled terminal |
| 13 | conflict | target exists / concurrent writer / failed replacement kept the working state |
| 14 | capacity | no proven plan fits BELOW the physical floor — quantified shortfall (needed N, had M, short by N−M for X); fires only after the ladder's deepest authorized rung, never exit 6, never a silent 0 |

Job terminals map onto it: succeeded 0 · failed 11 · canceled 12 · deadline 10.
