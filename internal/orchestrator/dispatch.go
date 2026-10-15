package orchestrator

import (
	"encoding/base64"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/machineendpoint"
	"github.com/cozy-creator/cozy/internal/records"
)

// Submission is one local request. The orchestrator owns everything in it that decides
// WHAT runs; the runtime owns everything about HOW.
type Submission struct {
	// Hub is the Tensorhub origin the request belongs to; empty is Calcifer's default.
	Hub                      string
	RequestID                string // server-reserved identity for request-owned input capture
	MachineExecutionObserver bool
	MachineEndpoint          *machineendpoint.Endpoint
	// TimeoutMS is the caller's --timeout; DeadlineUnixMS is when its run is canceled.
	TimeoutMS      int64
	DeadlineUnixMS uint64
	IdemKey        string // the caller's idempotency key
	Package        string // org/name
	Entrypoint     string // the function
	PlanID         string // the entrypoint_binding_plan_id this attempt binds
	Release        string // explicit published release; empty lets the machine select it
	// LocalInstallationID is the exact staged wheel-set identity for one editable rental.
	LocalInstallationID string
	Models              []ModelRef

	// Payload is the request body, verbatim. It rides the DeliveryGrant as the input
	// `payload` — a grant input, never a wire field, so refreshing the grant can never
	// substitute it.
	Payload []byte
	// Assets are immutable, Calcifer-staged files bound to exact payload field paths.
	// The request row keeps them so every requeue derives the same spec and grant.
	Assets []records.AssetBinding

	// Outputs is one destination per RESULT FIELD PATH (`image`, `detail.thumb`). Binding
	// by field path with exact set equality is what makes a two-output result
	// unswappable; a positional grant would silently cross them (decisions #248).
	Outputs []string

	// BodyDigest is the caller's own digest of the WHOLE submission it is making
	// idempotent, not merely of the payload. cl-006 supplies the digest of
	// (package, function, input, outputs) so that one key naming a different PACKAGE
	// conflicts as loudly as one naming different input — a digest over the payload
	// alone would let a key be reused across functions and mean two different things.
	// Empty falls back to the payload's digest.
	BodyDigest string

	// Kind is a function's call ("") or an operation's kind.
	Kind string
	// Trees are the typed input TREES as `ref=dir`, captured before the request is recorded.
	Trees []string
	// NeedsAccelerator is the function's declared machine class: false only for
	// `accelerator=False`. It is the only machine-class decision retained on a request.
	NeedsAccelerator bool
	// WritesWeights says the function declares weights outputs.
	WritesWeights bool

	// Warnings are recorded with the request, one request.warning event each.
	Warnings []records.Warning

	// Worker pins this request to an ATTACHED remote worker (a rental id resolved
	// through Options.Rentals). Empty = any local worker.
	Worker          string
	RequestedRental string
	// InstallID pins a durable request to one immutable local install resolution.
	// Remote requests instead carry their immutable Release.
	InstallID string
	// Rental authorizes placement on Creator-managed rented capacity.
	Rental bool
	// RentalRequired is the explicit development/E2E override that forbids local capacity.
	// It implies Rental and survives queue/restart scheduling in the request row.
	RentalRequired bool
	// RentNew requires an acquisition owned by this request, never an existing fleet rental.
	RentNew bool
	// PlannedSourceBytes is what a script ingest will pull; a rental bought for it is
	// sized to it.
	PlannedSourceBytes int64
	// OutputDirectory is the caller's explicit --out; empty means the package's store.
	// It is part of the submission's identity, where the derived intent below is not.
	OutputDirectory string
	// AttentionKernel is an optional developer execution-path pin. It affects only the
	// InvocationSpec and is intentionally excluded from placement and model resolution.
	AttentionKernel string
	// GPUs is zero for automatic package-supported parallelism or an exact user count.
	GPUs uint32
	// OutputExport is the derived publication obligation: the directory (explicit or
	// default) and the result-file contract. It changes no execution fact and is settled
	// independently after the terminal mirror.
	OutputExport *records.OutputExportIntent
	// ModelTransfer is the weights destination: the requester's repository the machine
	// publishes the function's weights outputs to. Nil keeps them on the machine.
	ModelTransfer *records.ModelTransferIntent
}

const WeightsManifestMime = "application/vnd.cozy.model-manifest"

// Result is what one closed attempt produced.
type Result struct {
	RequestID  string
	Attempt    uint64
	AttemptKey string
	Status     string
	Cause      string
	Outputs    []records.Output
	Body       []byte // the TerminalBody document, exactly as it was digested
}

// ActivateRecordedRequest is the second half for callers that transactionally
// attach request sidecars before ordinary queue activation.
func (c *Orchestrator) ActivateRecordedRequest(req records.Request) (uint64, *exit.Error) {
	return c.activateRecorded(req)
}

// RecordSubmission crosses the durable ordinary-request boundary.
func (c *Orchestrator) RecordSubmission(s Submission) (records.Request, bool, *exit.Error) {
	req, event, e := requestRecord(s)
	if e != nil {
		return records.Request{}, false, e
	}
	if req.Hub == "" {
		req.Hub = c.opt.Cfg.HubURL
	}
	// A run that names its machine is placed in the commit its submission waits on: this
	// computer's, or a named rental (refused there when released). Only a rental the fleet has
	// yet to choose is placed later.
	if req.Worker == "" {
		req.Worker = req.RequestedRental
	}
	if req.Placement = req.Worker; !req.Rental && !machineendpoint.IsName(req.Worker) {
		req.Placement = records.LocalMachine
	}
	req, fresh, e := c.opt.Store.SubmitWithEvent(req, event)
	if e != nil {
		return records.Request{}, false, e
	}
	if !fresh {
		c.logRecordedReplay(req, s.IdemKey)
		return req, false, nil
	}
	return req, true, nil
}

func requestRecord(s Submission) (records.Request, map[string]any, *exit.Error) {
	if s.RentNew {
		if s.RequestedRental != "" {
			return records.Request{}, nil, exit.Usagef("a fresh rental cannot reuse a selected machine")
		}
		s.RentalRequired = true
	}
	if s.RequestedRental != "" {
		if s.Worker != "" && s.Worker != s.RequestedRental {
			return records.Request{}, nil, exit.New(exit.Conflict, "assigned rental differs from requested rental")
		}
		s.RentalRequired = true
	}
	if s.RentalRequired {
		s.Rental = true
	}
	bodyDigest := s.BodyDigest
	if bodyDigest == "" {
		identity := s.Payload
		if s.Rental {
			document := map[string]canonical.Value{
				"payload": base64.StdEncoding.EncodeToString(s.Payload),
				"rental":  true, "rental_required": s.RentalRequired,
			}
			if s.RentNew {
				document["rent_new"] = true
			}
			if s.RequestedRental != "" {
				document["requested_rental"] = s.RequestedRental
			}
			encoded, err := canonical.Write(document)
			if err != nil {
				return records.Request{}, nil, exit.Internalf("cannot encode request budget identity: %s", err)
			}
			identity = encoded
		}
		spelled, err := canonical.Spell(canonical.Digest(identity))
		if err != nil {
			return records.Request{}, nil, exit.Internalf("cannot digest the request body: %s", err)
		}
		bodyDigest = spelled
	}
	if s.GPUs > 0 {
		identity, err := canonical.Write(map[string]canonical.Value{"body_digest": bodyDigest, "gpus": int64(s.GPUs)})
		if err != nil {
			return records.Request{}, nil, exit.Internalf("cannot encode GPU count identity: %s", err)
		}
		bodyDigest, err = canonical.Spell(canonical.Digest(identity))
		if err != nil {
			return records.Request{}, nil, exit.Internalf("cannot digest GPU count identity: %s", err)
		}
	}
	if s.AttentionKernel != "" {
		identity, err := canonical.Write(map[string]canonical.Value{
			"body_digest":      bodyDigest,
			"attention_kernel": s.AttentionKernel,
		})
		if err != nil {
			return records.Request{}, nil, exit.Internalf("cannot encode pinned request identity: %s", err)
		}
		bodyDigest, err = canonical.Spell(canonical.Digest(identity))
		if err != nil {
			return records.Request{}, nil, exit.Internalf("cannot digest pinned request identity: %s", err)
		}
	}
	if s.LocalInstallationID != "" {
		if s.InstallID == "" || s.LocalInstallationID == "" {
			return records.Request{}, nil, exit.Named(exit.Structural,
				"local_package_request_invalid",
				"a local package request requires its editable installation")
		}
		identity, err := canonical.Write(map[string]canonical.Value{
			"body_digest":           bodyDigest,
			"local_installation_id": s.LocalInstallationID,
		})
		if err != nil {
			return records.Request{}, nil, exit.Internalf(
				"cannot encode the local package request identity: %s", err)
		}
		bodyDigest, err = canonical.Spell(canonical.Digest(identity))
		if err != nil {
			return records.Request{}, nil, exit.Internalf(
				"cannot digest the local package request identity: %s", err)
		}
	}
	id := s.RequestID
	if id == "" {
		id = records.NewID("req")
	}
	req := records.Request{
		MachineEndpoint:          s.MachineEndpoint,
		MachineExecutionObserver: s.MachineExecutionObserver,
		DeadlineUnixMS:           s.DeadlineUnixMS,
		ID:                       id, IdemKey: s.IdemKey, BodyDigest: bodyDigest, Hub: s.Hub,
		Package: s.Package, Entrypoint: s.Entrypoint, PlanID: s.PlanID, Payload: s.Payload,
		Release:             s.Release,
		LocalInstallationID: s.LocalInstallationID,
		Outputs:             strings.Join(s.Outputs, ","),
		Assets:              s.Assets,
		Kind:                s.Kind, NeedsAccelerator: s.NeedsAccelerator, WritesWeights: s.WritesWeights,
		RequestedRental: s.RequestedRental,
		AttentionKernel: s.AttentionKernel, GPUs: s.GPUs,
		Worker: s.Worker, InstallID: s.InstallID, Rental: s.Rental,
		RentalRequired: s.RentalRequired, RentNew: s.RentNew, Models: s.Models,
		OutputExport: s.OutputExport, ModelTransfer: s.ModelTransfer, PlannedSourceBytes: s.PlannedSourceBytes,
		Warnings: s.Warnings,
	}
	event := map[string]any{
		"package": s.Package, "function": s.Entrypoint,
		"body_digest": bodyDigest, "plan_id": s.PlanID, "outputs": s.Outputs,
	}
	if s.ModelTransfer != nil {
		event["weights_destination"] = s.ModelTransfer.Destination
	}
	if s.MachineEndpoint != nil {
		event["machine_endpoint"] = s.MachineEndpoint
	}
	if s.TimeoutMS > 0 {
		event["timeout_ms"] = s.TimeoutMS
		event["deadline_unix_ms"] = s.DeadlineUnixMS
	}
	if s.GPUs > 0 {
		event["gpus"] = s.GPUs
	}
	if s.AttentionKernel != "" {
		event["attention_kernel"] = s.AttentionKernel
	}
	if s.Rental {
		event["rental"] = true
		event["rental_required"] = s.RentalRequired
		event["rent_new"] = s.RentNew
		event["release"] = s.Release
		if s.LocalInstallationID != "" {
			event["local_installation_id"] = s.LocalInstallationID
		}
	}
	return req, event, nil
}

func (c *Orchestrator) logRecordedReplay(req records.Request, idempotencyKey string) {
	// The recorded answer. A settled request is settled; a live one is already
	// running the attempt this call would otherwise duplicate.
	c.logf("request %s is the recorded answer for idempotency key %s (state %s, attempt %d)",
		req.ID, idempotencyKey, req.State, req.Ordinal)
}

func (c *Orchestrator) activateRecorded(req records.Request) (uint64, *exit.Error) {
	current, problem := c.opt.Store.RequestRow(req.ID)
	if problem != nil || current == nil {
		return 0, problem
	}
	if link, problem := c.opt.Store.MachineExecution(req.ID); problem != nil {
		return 0, problem
	} else if link != nil {
		if c.opt.StartMachineExecution == nil {
			return 0, exit.Unavailablef("this client cannot reconnect its Runtime-owned execution")
		}
		return 0, c.startMachineExecution(*current)
	}
	if current.State != "submitted" && current.State != "queued" {
		return uint64(current.Ordinal), nil
	}
	c.retireClassic(*current)
	return 0, ClassicRetired()
}

// startMachineExecution admits the async observer before releasing the closing
// fence. A delayed activation after HTTP admission may not cross a later down.
func (c *Orchestrator) startMachineExecution(req records.Request) *exit.Error {
	c.mu.Lock()
	if c.closing {
		c.mu.Unlock()
		return exit.Named(exit.Unavailable, "calcifer.closing", "Calcifer is closing; retained work will reconnect on startup")
	}
	token := "machine/" + req.ID
	if c.starting[token] {
		c.mu.Unlock()
		return nil
	}
	c.starting[token] = true
	c.mu.Unlock()
	defer func() { c.mu.Lock(); delete(c.starting, token); c.mu.Unlock() }()
	return c.opt.StartMachineExecution(req)
}

// start begins owed work: a machine execution on its machine; anything else was
// accepted for the retired classic worker and ends, named.
func (c *Orchestrator) start(req records.Request) {
	c.mu.Lock()
	closing := c.closing
	c.mu.Unlock()
	if closing {
		return
	}
	if link, problem := c.opt.Store.MachineExecution(req.ID); problem != nil || link != nil {
		if problem == nil && c.opt.StartMachineExecution != nil {
			_ = c.startMachineExecution(req)
		}
		return
	}
	current, problem := c.opt.Store.RequestRow(req.ID)
	if problem != nil || current == nil || (current.State != "submitted" && current.State != "queued") {
		return
	}
	c.retireClassic(*current)
}

// AwaitRental says why a queued request waits while the fleet buys its machine.
func (c *Orchestrator) AwaitRental(requestID, machine string) {
	req, problem := c.opt.Store.RequestRow(requestID)
	position := c.QueuePosition(requestID)
	if problem != nil || req == nil || position == 0 {
		return
	}
	reason := "waiting for rental " + machine + " to become ready"
	c.mu.Lock()
	changed := c.parked[requestID] != reason
	c.parked[requestID] = reason
	c.mu.Unlock()
	if !changed {
		return
	}
	c.logf("%s PARKED at queue position %d: %s", requestID, position, reason)
	c.emit(requestID, "request.parked", 0, waitFacts{cause: WaitRental, on: machine}.decorate(map[string]any{
		"reason": reason, "position": position,
	}, *req))
}

// requestSlot names what a request needs resident before assignment. Explicit
// rental affinity already identifies an independent machine at this point.
func requestSlot(req records.Request) string {
	slot := pinnedPackage(req.Package, req.RequestedRental)
	if req.RentNew {
		slot += "/fresh/" + req.ID
	}
	if req.InstallID != "" {
		slot += "/install/" + req.InstallID
	}
	return slot
}

// LogPlacement is the decision log for the capacity half: no worker held the placement,
// so the fleet placed the run on an attached rental or a bought pod (cl-165), and this
// request is pinned there until its placement reports.
func (c *Orchestrator) LogPlacement(req records.Request, decision PlacementDecision) {
	c.logf("%s: %s; candidates: %s", req.ID, decision.Line(), decision.verdicts())
	c.emit(req.ID, "request.placement", 0, decision.payload())
}

// settledState answers whether the authority has already recorded this request's outcome.
func settledState(state string) bool {
	switch state {
	case "succeeded", "failed", "canceled", "refused", "abandoned":
		return true
	}
	return false
}

// failQueued settles a request that can never be placed. It is a request-level terminal:
// no offer crossed to a worker, so there is no worker terminal to replay and the request
// row is what settles. A closed dispatch_aborted row may remain as preparation history.
// retireClassic ends a request no machine executes and that is not Calcifer's own
// transfer: it was accepted for the retired classic worker.
func (c *Orchestrator) retireClassic(req records.Request) {
	cause := ClassicRetired()
	applied, problem := c.opt.Store.FailQueuedPreparation(req, records.QueuedFailure(cause))
	if problem != nil || !applied {
		return
	}
	c.forget(req.ID)
	// An output obligation exists before attempt one. Settle it as skipped so
	// the caller does not wait for bytes that will never exist.
	c.RetryOutputExport(req.ID)
	go c.cleanupRequestAssets(req)
	c.logf("%s FAILED before any offer: %s", req.ID, cause.Message)
}

// DefaultMaxOutputMiB is the per-output bound when Options.MaxOutputMiB is unset. It
// matches the Runtime/package media object envelope; attempt and pod quotas still bound
// the aggregate.
const DefaultMaxOutputMiB int64 = 512

// ClientCancelGraceMS is the one cooperative attempt-cancellation policy. It is a
// cancellation budget carried to Runtime, never a stall or caller deadline.
const ClientCancelGraceMS uint64 = 5000
