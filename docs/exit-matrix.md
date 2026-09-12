# Error vocabulary

This table is the detailed domain and HTTP refusal vocabulary. `scripts/fence.py` checks
`internal/exit/exit.go` against it row for row. The CLI preserves each symbolic name in its
structured error document, but deliberately projects shell exits onto three outcomes:

- `0`: success or idempotent no-op.
- `2`: invocation or configuration error.
- `1`: every operational failure, including a failed/canceled invocation terminal.

| code | name | meaning |
|---|---|---|
| 0 | ok | success, including idempotent no-ops |
| 1 | internal | unexpected fault — a bug, never a user condition |
| 2 | usage | bad invocation: unknown flag, malformed `key=value`, malformed target |
| 3 | validation | typed payload/schema/bounds refusal; verifier refusal at ingest |
| 4 | not_found | unknown ref/function/package/attempt; hub 404 rendered verbatim |
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

The numeric values above remain useful inside the API/domain boundary and for HTTP status
projection; they are not the CLI process's exit status except for `ok`, `internal`, and `usage`.
