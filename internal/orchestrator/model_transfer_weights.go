package orchestrator

import (
	"context"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
)

type WeightsTransferDecision struct {
	ObjectID      string            `json:"object_id"`
	Length        int64             `json:"length"`
	URL           string            `json:"url,omitempty"`
	Headers       map[string]string `json:"headers,omitempty"`
	ExpiresAtUnix uint64            `json:"expires_at_unix,omitempty"`
	Held          bool              `json:"held,omitempty"`
}

// WeightsGrantMint is one mint: the decisions it authorized, and the SIGNER's own clock at
// the moment it signed them. Both halves are needed. An expiry alone cannot be compared to
// anything -- only `ExpiresAtUnix - ServerTimeUnix` states the life the hub actually signed
// for, and only the hub can state both ends of it.
type WeightsGrantMint struct {
	Decisions      []WeightsTransferDecision
	ServerTimeUnix int64
}

// WeightsGrantMinter authorizes the objects whose bytes are about to move. It is handed to
// the mover instead of a finished decision list precisely so that a grant is minted where it
// is spent: the mover walks its own outstanding set and asks for a window ahead of its
// cursor, so one signature's life bounds one object's start rather than a whole queue.
type WeightsGrantMinter func(context.Context, []string) (WeightsGrantMint, *exit.Error)

// grantWindowObjects is the hub's own per-call bound on a grant request.
const grantWindowObjects = 128

// WeightsGrantWindow holds the grants minted nearest the walk's cursor and knows when they
// have aged. AGE IS MEASURED AS ELAPSED LOCAL TIME AGAINST A HUB-DECLARED LIFE, never by
// comparing a local instant to a remote one: a duration is skew-free, an absolute instant is
// not. The hub says how long it signed for; this measures how much of that has been spent.
//
// It is exported because it IS the mover's half of the grant contract -- the thing that
// decides when a held signature stops being worth sending -- and the product suite drives it
// against a real expiry-enforcing origin.
type WeightsGrantWindow struct {
	mint     WeightsGrantMinter
	held     map[string]WeightsTransferDecision
	mintedAt time.Time
	life     time.Duration
}

// NewWeightsGrantWindow opens a window over one minter.
func NewWeightsGrantWindow(mint WeightsGrantMinter) *WeightsGrantWindow {
	return &WeightsGrantWindow{mint: mint}
}

// Spendable answers with a grant for objectID that is worth sending. `ahead` is the walk's
// remaining set starting at this object, so a refill mints forward from the cursor rather
// than one call per object.
func (w *WeightsGrantWindow) Spendable(ctx context.Context, objectID string, ahead []string,
	now time.Time,
) (WeightsTransferDecision, *exit.Error) {
	// Held custody has no signed URL to expire; it lasts for this publication.
	if decision, held := w.held[objectID]; held && (decision.Held || !w.Stale(now)) {
		return decision, nil
	}
	window := ahead
	if len(window) > grantWindowObjects {
		window = window[:grantWindowObjects]
	}
	minted, problem := w.mint(ctx, window)
	if problem != nil {
		return WeightsTransferDecision{}, problem
	}
	w.held = make(map[string]WeightsTransferDecision, len(minted.Decisions))
	for _, decision := range minted.Decisions {
		w.held[decision.ObjectID] = decision
	}
	w.mintedAt, w.life = now, minted.DeclaredLife()
	decision, held := w.held[objectID]
	if !held {
		return WeightsTransferDecision{}, exit.Internalf(
			"the hub authorized %d of %d asked objects without %s",
			len(minted.Decisions), len(window), objectID)
	}
	return decision, nil
}

// Stale is half the life the hub declared. It buys one stated guarantee: every object starts
// its transfer holding at least half a grant. That guarantee -- not the size of the number --
// is what makes ONE bounded signature lifetime correct for a walk of any size.
func (w *WeightsGrantWindow) Stale(now time.Time) bool {
	return w.life <= 0 || now.Sub(w.mintedAt) >= w.life/2
}

// DeclaredLife is the SHORTEST life in the mint, so the refill is driven by the first grant
// that will go cold rather than the last.
func (g WeightsGrantMint) DeclaredLife() time.Duration {
	shortest := int64(0)
	for _, decision := range g.Decisions {
		if decision.Held {
			continue // an object a concurrent publisher already accepted carries no signature
		}
		life := int64(decision.ExpiresAtUnix) - g.ServerTimeUnix
		if life <= 0 {
			return 0
		}
		if shortest == 0 || life < shortest {
			shortest = life
		}
	}
	return time.Duration(shortest) * time.Second
}

// Expire drops the window so the next send mints instead of re-presenting what it holds.
// Re-sending a URL that would not spend is a lie about time; being authorized again is not.
func (w *WeightsGrantWindow) Expire() { w.held, w.life = nil, 0 }
