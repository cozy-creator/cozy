package orchestrator

import (
	"encoding/base64"
	"encoding/json"
	"sort"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/machineendpoint"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// Submission is one local request. The orchestrator owns everything in it that decides
// WHAT runs; the runtime owns everything about HOW.
type Submission struct {
	// Hub is the Tensorhub origin the request belongs to; empty is the daemon's default.
	Hub                      string
	RequestID                string // server-reserved identity for request-owned input capture
	AllowPublish             []string
	MachineExecutionObserver bool
	MachineEndpoint          *machineendpoint.Endpoint
	TimeoutMS                int64
	DeadlineUnixMS           uint64
	IdemKey                  string // the caller's idempotency key
	Package                  string // org/name
	Entrypoint               string // the function
	PlanID                   string // the entrypoint_binding_plan_id this attempt binds
	Release                  string // immutable remote package release; empty for local execution
	// LocalInstallationID is the exact staged wheel-set identity for one editable rental.
	LocalInstallationID string
	Models              []ModelRef

	// Payload is the request body, verbatim. It rides the DeliveryGrant as the input
	// `payload` — a grant input, never a wire field, so refreshing the grant can never
	// substitute it.
	Payload []byte
	// Assets are immutable, daemon-staged files bound to exact payload field paths.
	// The request row keeps them so every requeue derives the same spec and grant.
	Assets []records.AssetBinding

	// Outputs is one destination per RESULT FIELD PATH (`image`, `detail.thumb`). Binding
	// by field path with exact set equality is what makes a two-output result
	// unswappable; a positional grant would silently cross them (decisions #248).
	Outputs []string
	// WeightsOutputs is the explicit Runtime-authored WeightsSink subset. Rev5's generic
	// OutputBinding carries no kind, so this is persisted beside the InvocationSpec and is
	// never inferred from an arriving receipt. The M0 lane is weights-only: when non-empty,
	// this set is the complete output set for the job.
	WeightsOutputs []WeightsOutput
	ProducerParams []string

	// BodyDigest is the caller's own digest of the WHOLE submission it is making
	// idempotent, not merely of the payload. cl-006 supplies the digest of
	// (package, function, input, outputs) so that one key naming a different PACKAGE
	// conflicts as loudly as one naming different input — a digest over the payload
	// alone would let a key be reused across functions and mean two different things.
	// Empty falls back to the payload's digest.
	BodyDigest string

	// Kind is the ATTEMPT CLASS: "" or `serving`, or `job`. A job carries two more facts
	// a serving request has no version of.
	Kind                string
	RetainWork          bool
	ReleaseImplicitWork bool
	RetryOf             string
	ChildReusable       bool
	ChildArtifacts      bool
	// Org is the publishing org whose scratch repo this job publishes into.
	Org string
	// Trees are the job's typed input TREES as `ref=dir`, one grant input each.
	Trees []string
	// NeedsAccelerator is derived from the selected package's immutable dependency facts.
	// It is the only machine-class decision retained on a request.
	NeedsAccelerator bool

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
	// OutputExport is the derived publication obligation: the directory (explicit or
	// default) and the result-file contract. It changes no execution fact and is settled
	// independently after the terminal mirror.
	OutputExport *records.OutputExportIntent
	// ModelTransfer is a privileged materializer/finalizer attached to this
	// ordinary request. It changes no lifecycle, placement, attempt, or event fact.
	ModelTransfer *records.ModelTransferIntent
}

const WeightsManifestMime = "application/vnd.cozy.model-manifest"

// WeightsOutput is one bounded WeightsSink slot projected from the installed job
// descriptor. MaxBytes bounds only newly written table/config bytes, not inherited closure.
type WeightsOutput struct {
	OutputID string `json:"output_id"`
	MimeType string `json:"mime_type"`
	MaxBytes uint64 `json:"max_bytes"`
}

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

// Submit records the request and dispatches its FIRST attempt. It is idempotent in the
// strong sense: the same key with the same body answers with the recorded request and
// its current attempt, and never starts a second execution. Re-dispatch is the
// orchestrator's requeue PROJECTION over a terminal, never a client repeating itself.
func (c *Orchestrator) Submit(s Submission) (string, uint64, *exit.Error) {
	id, attempt, _, e := c.SubmitDetail(s)
	return id, attempt, e
}

// SubmitDetail is Submit plus the one fact an HTTP host must not guess: whether THIS
// call started the work. A client that retried a timed-out POST needs "202, I started it"
// and "200, this key was already yours" to be different answers, and inferring it from
// equal request ids is a race.
func (c *Orchestrator) SubmitDetail(s Submission) (string, uint64, bool, *exit.Error) {
	req, fresh, e := c.RecordSubmission(s)
	if e != nil {
		return "", 0, false, e
	}
	if !fresh {
		return req.ID, uint64(req.Ordinal), false, nil
	}
	attempt, e := c.activateRecorded(req)
	return req.ID, attempt, true, e
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
		if s.RequestedRental != "" || s.RetryOf != "" {
			return records.Request{}, nil, exit.Usagef("a fresh rental cannot reuse a selected machine or retained run")
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
	if s.RetainWork && s.Kind != "job" {
		return records.Request{}, nil, exit.Named(exit.Validation, "retain_work_not_job",
			"retained work is an ordinary unpublished package job capability")
	}
	weightsOutputs, weightsBytes, e := normalizeWeightsOutputs(s)
	if e != nil {
		return records.Request{}, nil, e
	}
	s.WeightsOutputs = weightsOutputs
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
		if s.Kind == "job" {
			id = records.NewID("job")
		}
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
		Assets:              s.Assets, WeightsOutputs: string(weightsBytes),
		Kind: s.Kind, RetainWork: s.RetainWork, ReleaseImplicitWork: s.ReleaseImplicitWork, RetryOf: s.RetryOf, ChildArtifacts: s.ChildArtifacts, NeedsAccelerator: s.NeedsAccelerator, Org: s.Org, Trees: strings.Join(s.Trees, ","),
		RequestedRental: s.RequestedRental,
		AttentionKernel: s.AttentionKernel,
		Worker:          s.Worker, InstallID: s.InstallID, Rental: s.Rental,
		RentalRequired: s.RentalRequired, RentNew: s.RentNew, Models: s.Models,
		OutputExport: s.OutputExport, ModelTransfer: s.ModelTransfer, PlannedSourceBytes: s.PlannedSourceBytes,
		Warnings: s.Warnings,
	}
	event := map[string]any{
		"retain_work": s.RetainWork,
		"package":     s.Package, "function": s.Entrypoint,
		"body_digest": bodyDigest, "plan_id": s.PlanID, "outputs": s.Outputs,
		"weights_outputs": weightsOutputs,
	}
	if s.MachineEndpoint != nil {
		event["machine_endpoint"] = s.MachineEndpoint
	}
	if s.TimeoutMS > 0 {
		event["timeout_ms"] = s.TimeoutMS
		event["deadline_unix_ms"] = s.DeadlineUnixMS
	}
	if len(s.AllowPublish) > 0 {
		event["allow_publish"] = s.AllowPublish
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

func normalizeWeightsOutputs(s Submission) ([]WeightsOutput, []byte, *exit.Error) {
	if len(s.WeightsOutputs) == 0 {
		return nil, []byte("[]"), nil
	}
	if s.Kind != "job" {
		return nil, nil, exit.Named(exit.Validation, "weights_output_not_job",
			"weights outputs are valid only on a job submission")
	}
	if len(s.WeightsOutputs) > pb.MaxWeightsReceipts {
		return nil, nil, exit.Named(exit.Validation, "weights_output_count_cap",
			"%d weights outputs exceeds the protocol cap of %d",
			len(s.WeightsOutputs), pb.MaxWeightsReceipts)
	}
	rows := append([]WeightsOutput(nil), s.WeightsOutputs...)
	sort.Slice(rows, func(i, j int) bool { return rows[i].OutputID < rows[j].OutputID })
	ids := make(map[string]bool, len(rows))
	for _, row := range rows {
		if row.OutputID == "" || ids[row.OutputID] {
			return nil, nil, exit.Named(exit.Validation, "weights_output_identity",
				"weights output slots are non-empty and unique; %q is repeated or empty", row.OutputID)
		}
		if row.MimeType != WeightsManifestMime || row.MaxBytes > (uint64(1)<<53)-1 {
			return nil, nil, exit.Named(exit.Validation, "weights_output_contract",
				"weights output %s must declare MIME %s and a new-byte cap in 0..2^53-1",
				row.OutputID, WeightsManifestMime)
		}
		ids[row.OutputID] = true
	}
	// OutputBinding/1 has no kind. Until that schema gap is closed, an WeightsSink job is
	// weights-only so a missing receipt can be classified without guessing about asset slots.
	if len(ids) != len(s.Outputs) {
		return nil, nil, exit.Named(exit.Structural, "mixed_job_output_kinds",
			"this rev5 lane requires weights-only jobs; %d weights slots do not close %d outputs",
			len(ids), len(s.Outputs))
	}
	for _, id := range s.Outputs {
		if !ids[id] {
			return nil, nil, exit.Named(exit.Structural, "mixed_job_output_kinds",
				"job output %q is not in the explicit weights-output set", id)
		}
	}
	data, err := json.Marshal(rows)
	if err != nil {
		return nil, nil, exit.Internalf("cannot persist weights output declarations: %s", err)
	}
	return rows, data, nil
}

func decodeWeightsOutputs(data string) ([]WeightsOutput, *exit.Error) {
	if data == "" {
		return nil, nil
	}
	var rows []WeightsOutput
	if err := json.Unmarshal([]byte(data), &rows); err != nil {
		return nil, exit.Internalf("cannot decode persisted weights output declarations: %s", err)
	}
	return rows, nil
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
	if passThrough(*current) {
		// An unchanged verified Manifest has no code to execute and no bytes to
		// reproduce; the daemon moves it and settles the ordinary request.
		go c.runModelPassThrough(*current)
		return 0, nil
	}
	c.retireClassic(*current)
	return 0, ClassicRetired()
}

// passThrough is the daemon's own model transfer: no machine executes it.
func passThrough(req records.Request) bool {
	return req.ModelTransfer != nil && req.Package == "cozy/platform" && req.Entrypoint == "model-pass-through"
}

// startMachineExecution admits the async observer before releasing the closing
// fence. A delayed activation after HTTP admission may not cross a later down.
func (c *Orchestrator) startMachineExecution(req records.Request) *exit.Error {
	c.mu.Lock()
	if c.closing {
		c.mu.Unlock()
		return exit.Named(exit.Unavailable, "daemon.closing", "the daemon is closing; retained work will reconnect on startup")
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
	if passThrough(req) {
		go c.runModelPassThrough(req)
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
	if req.IsJob() {
		slot += "/job/" + req.Entrypoint
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

// selectionServes is the model half of the match (cl-114): a placement that holds a slot
// the request binds must hold it under the request's exact manifest. The plan id hashes
// the entrypoint's interface, not its weights, so two selections of one package share a
// plan — matching on the plan alone dispatched an fp8 request onto the warm bf16
// placement. A request that binds no models (an editable install's frozen selection, an
// unmodeled package) accepts whatever the placement holds, and a placement whose set
// binds no row for a slot has no selection to disagree with — the request's model rows
// are then facts for the row, not residency evidence.
func selectionServes(requested, held []ModelRef) bool {
	if len(requested) == 0 || len(held) == 0 {
		return !hasModelAdapters(requested)
	}
	holds := make(map[string]ModelRef, len(held))
	for _, m := range held {
		holds[m.BindingSlot()] = m
	}
	for _, m := range requested {
		current, ok := holds[m.BindingSlot()]
		if !ok {
			if len(m.Adapters) > 0 {
				return false
			}
			continue
		}
		if !records.SameAdapters(m.Adapters, current.Adapters) {
			return false
		}
		if _, fits := rungHolding(m, current.Manifest); !fits {
			return false
		}
	}
	return true
}

// Readiness and the final offer both require facts from the worker's actual
// PlacementSet. A protocol minor or a copied request is not feature support.
func requireAdapterEcho(requested, observed []ModelRef) *exit.Error {
	if (hasModelAdapters(requested) || hasModelAdapters(observed)) && !selectionServes(requested, observed) {
		return exit.Named(exit.Structural, "model_adapters_preparation_mismatch",
			"worker preparation omitted or changed the requested LoRA stack")
	}
	return nil
}

// rungHolding answers whether a held manifest is one the request's ref accepts: its own
// pin, or — unpinned — any rung of its ladder (cl-166). The rung is what a dispatch onto
// that placement pins the request to.
func rungHolding(m ModelRef, manifest string) (records.ModelRung, bool) {
	if m.Pinned() {
		return records.ModelRung{GPUs: m.GPUs, Lane: m.Lane, Manifest: m.Manifest, Bytes: m.Bytes}, m.Manifest == manifest
	}
	for _, rung := range m.Ladder {
		if rung.Manifest == manifest {
			return rung, true
		}
	}
	return records.ModelRung{}, false
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
// retireClassic ends a request no machine executes and that is not the daemon's own
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
	c.signalClosed(requestWaitKey(req.ID), cause)
}

// DefaultMaxOutputMiB is the per-output bound when Options.MaxOutputMiB is unset. It
// matches the Runtime/package media object envelope; attempt and pod quotas still bound
// the aggregate.
const DefaultMaxOutputMiB int64 = 512

func spellOf(raw []byte) string {
	s, err := canonical.Spell(raw)
	if err != nil {
		return ""
	}
	return s
}

// inputBindings is the spec's ORDERED input identity list (#439): the payload always,
// a job's exact Model manifests, plus a job's materialized trees (their verification is
// the store's own, so no digest). A SERVING request never declares a Model input: its
// models arrive through the placement's package-set lane, and the worker refuses a
// serving spec that carries one (grant_model_serving_refused).
func inputBindings(req records.Request, payloadDigest string) []*pb.InputBinding {
	rows := []*pb.InputBinding{{
		InputId:  "payload",
		Digest:   payloadDigest,
		Length:   uint64(len(req.Payload)),
		KindMime: "application/json",
		Order:    0,
	}}
	for _, asset := range req.Assets {
		rows = append(rows, &pb.InputBinding{
			InputId: asset.FieldPath, Digest: asset.Digest, Length: uint64(asset.Length),
			KindMime: asset.MediaType, Order: asset.Order,
		})
	}
	models := jobModels(req)
	sort.Slice(models, func(i, j int) bool { return models[i].Slot < models[j].Slot })
	for _, model := range models {
		rows = append(rows, &pb.InputBinding{
			InputId: "model:" + model.Slot, Digest: model.Manifest,
			Length:   uint64(model.ManifestLength),
			KindMime: "application/vnd.cozy.model-manifest",
			Order:    0, // each Model is a scalar parameter, not an element of one shared list
		})
	}
	order := uint32(len(rows))
	for _, pair := range splitList(req.Trees) {
		ref, _, ok := strings.Cut(pair, "=")
		if !ok {
			continue
		}
		rows = append(rows, &pb.InputBinding{
			InputId: "tree:" + ref, KindMime: "inode/directory", Order: order,
		})
		order++
	}
	return rows
}

func downloadModelRefs(models []ModelRef) []*pb.DownloadModelRef {
	out := make([]*pb.DownloadModelRef, 0, len(models))
	for _, model := range models {
		// Operation-local manifests are already held by this worker. A retained
		// Hub checkpoint can be downloaded even when no release names it.
		if !model.Downloadable() {
			continue
		}
		// ONE PLACEMENT PER CONSTRUCTION (h3a-018): the selection rides under every slot
		// that shares its bytes, so the pod binds each of those entrypoints in the one
		// placement it prepares.
		for _, slot := range append([]string{model.BindingSlot()}, model.SharedSlots...) {
			out = append(out, &pb.DownloadModelRef{Package: model.Package, Slot: slot,
				Model: model.Model, Release: model.Release, Lane: model.Lane, Manifest: model.Manifest, Adapters: downloadAdapters(model.Adapters)})
		}
	}
	return out
}

// DownloadModelRefs projects the exact model bindings captured for a published
// request onto the worker's package preparation contract.  Keeping this
// projection in the orchestrator ensures every preparation path (ordinary run,
// explicit prefetch, and recovery) carries the same slot, manifest, and adapter
// identity to Runtime.
func DownloadModelRefs(models []ModelRef) []*pb.DownloadModelRef {
	return downloadModelRefs(models)
}

func invocationOutputBindings(ids []string, weights []WeightsOutput, defaultMax uint64) []*pb.OutputBinding {
	byID := map[string]WeightsOutput{}
	for _, output := range weights {
		byID[output.OutputID] = output
	}
	out := make([]*pb.OutputBinding, 0, len(ids))
	for _, id := range ids {
		binding := &pb.OutputBinding{OutputId: id, MaxBytes: defaultMax}
		if weights, ok := byID[id]; ok {
			binding.MimeType = weights.MimeType
			binding.MaxBytes = weights.MaxBytes
		}
		out = append(out, binding)
	}
	return out
}

// jobModels is the set of Model inputs a request declares: a job's bound models with an
// exact manifest; none for a serving request (see inputBindings).
func jobModels(req records.Request) []ModelRef {
	if !req.IsJob() {
		return nil
	}
	var models []ModelRef
	for _, model := range req.OwnModels() { // callee defaults are never the root's inputs
		if model.ManifestLength > 0 {
			models = append(models, model)
		}
	}
	return models
}

// modelAccess is the access half of every Model input the spec declares (jobModels):
// one row per model, the same on every grant, or the worker refuses
// grant_binding_mismatch.
func modelAccess(req records.Request) []*pb.InputAccess {
	var rows []*pb.InputAccess
	for _, model := range jobModels(req) {
		rows = append(rows, &pb.InputAccess{
			InputId: "model:" + model.Slot, Url: "model://" + model.Manifest,
		})
	}
	return rows
}

// Await blocks until the attempt is closed — the terminal accepted, its outputs visible,
// and the ack sent. The error it returns is the orchestrator's PROJECTION of the neutral
// terminal, never a worker-authored retryability claim.
func (c *Orchestrator) Await(requestID string, attempt uint64, timeout time.Duration) (*Result, *exit.Error) {
	w, release := c.acquireWait(key(requestID, attempt))
	defer release()
	if row, e := c.opt.Store.AttemptRow(requestID, int64(attempt)); e == nil && row != nil && row.State == "closed" {
		w.markClosed(outcomeError(row.TerminalStatus, row.TerminalCause, row.SafeMessage))
	}
	select {
	case <-w.closed:
	case <-time.After(timeout):
		return nil, exit.New(exit.Deadline, "%s#%d did not reach a terminal in %s",
			requestID, attempt, timeout)
	}
	row, e := c.opt.Store.AttemptRow(requestID, int64(attempt))
	if e != nil {
		return nil, e
	}
	if row == nil {
		return nil, exit.Internalf("%s#%d closed with no row", requestID, attempt)
	}
	outs, e := c.opt.Store.VisibleOutputs(requestID)
	if e != nil {
		return nil, e
	}
	return &Result{
		RequestID: requestID, Attempt: attempt, AttemptKey: row.AttemptKey,
		Status: row.TerminalStatus, Cause: row.TerminalCause,
		Outputs: outs, Body: row.TerminalBody,
	}, w.err
}

// AwaitSettled blocks until the REQUEST settles — through however many requeued
// ordinals the orchestrator's projection minted. This is what a caller waits on; an
// individual attempt is the orchestrator's business.
func (c *Orchestrator) AwaitSettled(requestID string, timeout time.Duration) (*Result, *exit.Error) {
	w, release := c.acquireWait(requestWaitKey(requestID))
	defer release()
	if row, e := c.opt.Store.RequestRow(requestID); e == nil && row != nil && settledState(row.State) {
		attempts, read := c.opt.Store.Attempts(requestID)
		if read == nil && len(attempts) > 0 {
			last := attempts[len(attempts)-1]
			w.markClosed(outcomeError(last.TerminalStatus, last.TerminalCause, last.SafeMessage))
		} else {
			w.markClosed(exit.Named(exit.Failed, row.State, "request %s settled %s before an attempt", requestID, row.State))
		}
	}
	select {
	case <-w.closed:
	case <-time.After(timeout):
		return nil, exit.New(exit.Deadline, "%s did not settle in %s", requestID, timeout)
	}
	row, e := c.opt.Store.RequestRow(requestID)
	if e != nil || row == nil {
		return nil, exit.Internalf("%s settled with no row", requestID)
	}
	attempts, e := c.opt.Store.Attempts(requestID)
	if e != nil || len(attempts) == 0 {
		return nil, exit.Internalf("%s settled with no attempt", requestID)
	}
	last := attempts[len(attempts)-1]
	outs, e := c.opt.Store.VisibleOutputs(requestID)
	if e != nil {
		return nil, e
	}
	return &Result{
		RequestID: requestID, Attempt: uint64(last.Attempt), AttemptKey: last.AttemptKey,
		Status: last.TerminalStatus, Cause: last.TerminalCause, Outputs: outs,
		Body: last.TerminalBody,
	}, w.err
}

// AwaitAccepted blocks until AttemptAccepted is journaled — the acceptance boundary a
// submit→accepted benchmark measures.
func (c *Orchestrator) AwaitAccepted(requestID string, attempt uint64, timeout time.Duration) *exit.Error {
	w, release := c.acquireWait(key(requestID, attempt))
	defer release()
	if row, e := c.opt.Store.AttemptRow(requestID, int64(attempt)); e == nil && row != nil {
		if row.AcceptedAt != "" {
			w.markAccepted()
		}
		if row.State == "closed" {
			w.markClosed(outcomeError(row.TerminalStatus, row.TerminalCause, row.SafeMessage))
		}
	}
	select {
	case <-w.accepted:
		return nil
	case <-w.closed:
		return w.err
	case <-time.After(timeout):
		return exit.New(exit.Deadline, "%s#%d was not accepted in %s", requestID, attempt, timeout)
	}
}

// ClientCancelGraceMS is the one cooperative attempt-cancellation policy. It is a
// cancellation budget carried to Runtime, never a stall or caller deadline.
const ClientCancelGraceMS uint64 = 5000
