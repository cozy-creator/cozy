package orchestrator

// THE RECORDOWNER SIDE of th-024's orientation (#436/#446/#454, renamed by #481 — bare
// "owner" is retired and the formal role is RecordOwner): the WORKER hosts `WorkerControl`
// and THIS side dials it, claims it, reconciles its ONE snapshot, and only then
// dispatches. One goroutine per worker owns the whole conversation; the `session` is that
// stream's sender half, fenced by the control stream epoch the worker minted.
//
// LAUNCH TIER (#437): record_owner_epoch is the constant 1 — this daemon is the one
// RecordOwner of every worker it spawns or connects to; the machinery that MINTS competing
// epochs is the hub's Wave-2 lease. Rented workers authenticate the channel with the
// provisioned Creator mTLS key.

const recordOwnerEpoch = 1

// recordOwnerID names this owner on Claim. Stable per daemon run is enough at launch:
// equal-epoch claims from the SAME RecordOwner are reconnects, anything else is refused.
const recordOwnerID = "cozy-local-client"

// RecordOwnerID and RecordOwnerEpoch are the owner every machine Claim names.
const (
	RecordOwnerID    = recordOwnerID
	RecordOwnerEpoch = recordOwnerEpoch
)
