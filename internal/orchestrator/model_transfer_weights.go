package orchestrator

import (
	"context"
	"encoding/base64"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
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
	if decision, held := w.held[objectID]; held && !w.Stale(now) {
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

// Expire drops the window so the next send mints instead of re-presenting what it holds.
// Re-sending a URL that would not spend is a lie about time; being authorized again is not.
func (w *WeightsGrantWindow) Expire() { w.held, w.life = nil, 0 }

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

func (c *Orchestrator) onModelTransferWeightsReceipt(s *session, frame *pb.WeightsReceiptFrame) {
	transfer, problem := c.opt.Store.ModelTransferOf(frame.RequestId)
	if problem != nil || transfer == nil {
		return
	}
	request, problem := c.opt.Store.RequestRow(frame.RequestId)
	if problem != nil || request == nil || !request.IsJob() {
		return
	}
	attempt, problem := c.opt.Store.AttemptRow(frame.RequestId, int64(frame.AttemptOrdinal))
	if problem != nil || attempt == nil || attempt.InstanceID != s.instanceID {
		return
	}
	invocationDigest, err := canonical.Spell(frame.InvocationSpecDigest)
	manifestDigest := ""
	if frame.Manifest != nil {
		manifestDigest, _ = canonical.Spell(frame.Manifest.Digest)
	}
	if err != nil || invocationDigest != attempt.InvocationDigest || frame.WeightsReceipt == nil ||
		frame.Manifest == nil || manifestDigest == "" || frame.Manifest.Length == 0 ||
		frame.WriterEpoch == 0 || frame.WeightsTransactionId == "" {
		return
	}
	receipt, problem := parseWeightsReceiptRef(canonical.Doc{
		"weights_receipt_digest": canonicalSpell(frame.WeightsReceipt.WeightsReceiptDigest),
		"weights_receipt_canonical_bytes": base64.StdEncoding.EncodeToString(
			frame.WeightsReceipt.WeightsReceiptCanonicalBytes),
	})
	if problem != nil || receipt.RequestID != frame.RequestId ||
		receipt.InvocationDigest != invocationDigest || receipt.OutputSlot != frame.OutputSlot {
		return
	}
	receiptDoc, err := canonical.Read(receipt.ReceiptBytes, &pb.WeightsReceipt{})
	if err != nil || receiptDoc.Str("weights_transaction_id") != frame.WeightsTransactionId {
		return
	}
	objects := make([]records.ModelTransferObject, 0, len(frame.Objects))
	previous := ""
	manifestPresent := false
	for _, object := range frame.Objects {
		if object == nil || object.ObjectId <= previous || object.Length == 0 ||
			object.Length > uint64(^uint64(0)>>1) || object.SourceRef == "" {
			return
		}
		if _, digestErr := canonical.Raw(object.ObjectId); digestErr != nil {
			return
		}
		previous = object.ObjectId
		manifestPresent = manifestPresent || object.ObjectId == manifestDigest &&
			object.Length == frame.Manifest.Length
		objects = append(objects, records.ModelTransferObject{ObjectID: object.ObjectId,
			Length: int64(object.Length), SourceRef: object.SourceRef})
	}
	if len(objects) == 0 || !manifestPresent {
		return
	}
	problem = c.opt.Store.RecordModelTransferWeights(records.ModelTransferWeights{
		RequestID: frame.RequestId, OutputSlot: frame.OutputSlot, ManifestID: manifestDigest,
		ManifestLength: int64(frame.Manifest.Length), Objects: objects,
		Attempt: int64(frame.AttemptOrdinal), InvocationDigest: invocationDigest,
		TransactionID: frame.WeightsTransactionId, ReceiptDigest: receipt.ReceiptDigest,
		Receipt: receipt.ReceiptBytes})
	if problem == nil {
		c.signalTransfer(frame.RequestId)
	}
}

func (c *Orchestrator) onModelTransferWeightsStatus(s *session,
	frame *pb.WeightsTransferStatus,
) {
	weights, problem := c.opt.Store.ModelTransferWeights(frame.RequestId,
		int64(frame.AttemptOrdinal), frame.OutputSlot)
	if problem != nil || weights == nil || frame.Length == 0 || frame.Length > uint64(^uint64(0)>>1) {
		return
	}
	invocation := canonicalSpell(frame.InvocationSpecDigest)
	if weights.Attempt != int64(frame.AttemptOrdinal) || weights.InvocationDigest != invocation ||
		weights.TransactionID != frame.WeightsTransactionId {
		return
	}
	request, problem := c.opt.Store.RequestRow(frame.RequestId)
	attempt, attemptProblem := c.opt.Store.AttemptRow(frame.RequestId, int64(frame.AttemptOrdinal))
	if problem != nil || attemptProblem != nil || request == nil || attempt == nil ||
		request.Worker == "" || rentalInstanceID(request.Worker) != s.instanceID ||
		attempt.InstanceID != s.instanceID {
		return
	}
	state := trimEnum(pb.WeightsTransferState_name[int32(frame.State)], "WEIGHTS_TRANSFER_STATE_")
	state = map[string]string{"ACCEPTED": "accepted", "READING": "reading",
		"UPLOADING": "uploading", "UPLOADED": "uploaded", "ALREADY_PRESENT": "already_present",
		"HELD": "held", "FAILED": "failed"}[state]
	if state == "" || frame.TransferredBytes > frame.Length {
		return
	}
	previous := ""
	if objects, readProblem := c.opt.Store.ModelTransferObjects(frame.RequestId,
		int64(frame.AttemptOrdinal), frame.OutputSlot); readProblem == nil {
		for _, object := range objects {
			if object.ObjectID == frame.ObjectId {
				previous = object.State
			}
		}
	}
	problem = c.opt.Store.RecordModelTransferObjectStatus(records.ModelTransferObject{
		RequestID: frame.RequestId, Attempt: int64(frame.AttemptOrdinal),
		OutputSlot: frame.OutputSlot, ObjectID: frame.ObjectId, Length: int64(frame.Length),
		OperationID: frame.OperationId, GrantRevision: int64(frame.GrantRevision),
		UpdateSequence: int64(frame.UpdateSequence), State: state,
		Transferred: int64(frame.TransferredBytes), SafeCode: frame.SafeCode,
		SafeDetail: frame.SafeDetail})
	if problem != nil {
		if state == "failed" || problem.ErrName() != "model_transfer.object_status_superseded" {
			c.logf("model transfer %s: object %s reported %s under grant revision %d/%d "+
				"(%s: %s) and the row did not take it: %s", frame.RequestId, frame.ObjectId,
				state, frame.GrantRevision, frame.UpdateSequence, frame.SafeCode,
				frame.SafeDetail, problem.Message)
		}
		return
	}
	// The same rule as the source lane: the wake is for facts the mover can act on, and
	// `moveModelTransferWeights` decides on an object's STATE. Waking on byte progress made
	// the pod's own echo re-send every non-adopted object at `grant_revision + 1` — the
	// restatement loop of cl-136 on the outbound half, and the manufacturer of the stale
	// grant revisions cl-133 was swallowing. The re-send itself stays: it is how an object
	// whose grant went cold gets a signature minted just now (`WeightsGrantWindow.Expire`),
	// and on a state change it is exactly what should happen.
	if state != previous {
		c.signalTransfer(frame.RequestId)
	}
}

func canonicalSpell(raw []byte) string {
	value, _ := canonical.Spell(raw)
	return value
}
