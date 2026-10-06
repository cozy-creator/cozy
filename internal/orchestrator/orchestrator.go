// Package orchestrator records this controller's request intentions and observations.
// Machines own execution attempts and input/output custody. The controller admits authored
// work, retains cancellation intent, observes native runs and reconciles its own outboxes.
// Observer teardown does not mutate a run.
package orchestrator

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"sync"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/records"
)

// Options is the frozen input to one Cozy daemon. Every field is decided by the
// entrypoint; nothing in this package reads the environment.
type Options struct {
	// StartMachineExecution transfers and observes an execution owned by Runtime.
	StartMachineExecution func(records.Request) *exit.Error
	// ReclaimInstall delegates unpinned snapshot cleanup to the existing package owner.
	ReclaimInstall func(string) *exit.Error
	Cfg            config.Config
	Layout         home.Layout
	Store          *records.Store
	Log            io.Writer
	// Rentals resolves an attached rental id to its dial spec; nil = this daemon attaches
	// no rentals.
	Rentals        func(id string) (*RemoteTarget, *exit.Error)
	ModelTransfers ModelTransferOwner
}

type RentalObservation struct {
	RentalID       string
	Accelerator    string
	DeviceCount    int
	Backend        string
	WorkerInstance string
	WorkerID       string
	WorkerBootID   string
}

// ModelTransferOwner moves the daemon's own model transfers.
type ModelTransferOwner interface {
	PassThrough(context.Context, string, records.ModelTransferIntent) *exit.Error
}

type ModelTransferMover func(context.Context, records.ModelTransferWeights,
	WeightsGrantMinter) *exit.Error

// Launcher resolves a package ref along the two boundaries #484 split: the
// platform-neutral desired placement and the local target-environment launch. A connected
// worker asks only for ResolvePlacement; it must never force this host to materialize or
// execute the target environment merely to author a remote plan.
type Launcher interface {
	ValidateExecutionCapture(records.Request) *exit.Error
	ResolvePlacement(pkg string) (DesiredPlacement, *exit.Error)
	LocalInstallation(installID, digest string) (localpackage.Installation, *exit.Error)
}

type LogicalPackage struct {
	Package          string
	Release          string
	Function         string
	Outputs          []string
	PlanID           string
	Models           []ModelRef
	NeedsAccelerator bool
}

type LogicalJob struct {
	Package          string
	Release          string
	Function         string
	DescriptorID     string
	Outputs          []string
	WeightsOutputs   []WeightsOutput
	NeedsAccelerator bool
	Models           []ModelRef
	ProducerParams   []string
}

type ModelRef = records.ModelRef

// PlacementDecision is the capacity decision's whole record (cl-165, placement-economics.md):
// the tier it ran under, the config it read, every throughput row it used, and every
// candidate — attached ready rental or purchasable product — with its rate, expected
// time, cost, score and verdict. It is the `request.placement` event. A record naming
// only its winner cannot be audited: on 2026-09-04 a `--rental-only` run bought a second
// pod while a ready rental sat idle, and the durable record could not say why.
type PlacementDecision struct {
	Tier         string `json:"tier"`
	ConfigDigest string `json:"config_digest"`
	// Ladder is the owner's fit map the candidates were sized by (`H100=fp8 > *=bf16`),
	// one entry per slot when slots differ; none when the request binds no ladder.
	// Override is the lane the request pinned itself — an explicit `model.<param>=…/lane`
	// (cl-170) — which is not a rung and asserts no fit.
	Ladder     []string              `json:"ladder,omitempty"`
	Override   string                `json:"override,omitempty"`
	Throughput []hub.ModelThroughput `json:"throughput"`
	Candidates []PlacementCandidate  `json:"candidates"`
	// RentalID is the machine chosen; empty on a decision to wait for one attaching.
	RentalID string `json:"rental,omitempty"`
	Bought   bool   `json:"bought"`
	// Models is the request's selection pinned to the chosen machine's rung; nil when
	// the request was already exact.
	Models []ModelRef `json:"-"`
}

// PlacementCandidate is one machine the run could be placed on, as the choice saw it: an
// attached ready rental (Rental set) or a purchasable product.
type PlacementCandidate struct {
	Rental  string `json:"rental,omitempty"`
	Machine string `json:"machine,omitempty"`
	SKU     string `json:"sku"`
	// GPUs is the machine's GPU count: the product width a buy asks for, or an attached
	// rental's own.
	GPUs int `json:"gpus"`
	// Rung is the 1-based ladder rung the machine fell under — 0 under an explicit lane,
	// which is not a rung — and Lane the lane it pins; Fit says how the device was sized
	// against that lane (cl-168, cl-170).
	Rung int    `json:"rung,omitempty"`
	Lane string `json:"lane,omitempty"`
	Fit  string `json:"fit,omitempty"`
	// Ahead is the attempts a new request waits behind on an attached rental.
	Ahead int `json:"ahead,omitempty"`
	// RateUSDMicrosPerHour is what the renter pays: the live catalog's price plus storage.
	RateUSDMicrosPerHour int64 `json:"rate_usd_micros_per_hour"`
	// Measured says a throughput row exists for (lane, sku, gpus). TimeS, CostUSDMicros and
	// Score follow placement-economics.md and are zero on an unmeasured candidate.
	Measured      bool    `json:"measured"`
	TimeS         float64 `json:"time_s,omitempty"`
	CostUSDMicros int64   `json:"cost_usd_micros,omitempty"`
	Score         float64 `json:"score,omitempty"`
	// DiskUnknown marks an attached rental whose Hub reported no disk while the request
	// needs one; it is chosen only after the rentals known to fit.
	DiskUnknown bool `json:"disk_unknown,omitempty"`
	// Verdict is one of the constants below; empty only while the choice is still open.
	Verdict string     `json:"verdict"`
	Models  []ModelRef `json:"-"`
}

// Verdicts a placement candidate carries once the choice is made.
const (
	VerdictChosen     = "chosen"
	VerdictSlower     = "slower"
	VerdictDearer     = "dearer"
	VerdictUnmeasured = "unmeasured"
	// VerdictAttaching: a fitting rental the fleet holds that cannot take the request
	// YET but is provably on its way — anywhere from `pending_acquisition` through a
	// `ready` pod whose worker has not attached. The request waits for it (cl-170) and
	// is never failed on it (cl-185).
	VerdictAttaching = "attaching"
	// VerdictNoRung: no rung of the binding ladder names this machine's accelerator.
	VerdictNoRung = "no_rung"
	// VerdictNoStock: the hub refused the buy for want of inventory, and the choice repeated.
	VerdictNoStock = "no_stock"
	// VerdictExcluded is a prefix; the reason follows (`excluded:vram_short: …`).
	VerdictExcluded = "excluded:"
)

// Attached says the candidate is a rental this daemon already holds.
func (c PlacementCandidate) Attached() bool { return c.Rental != "" }

// Name is how a reader knows the candidate: the machine word, or the product.
func (c PlacementCandidate) Name() string {
	if c.Machine != "" {
		return c.Machine
	}
	if c.Rental != "" {
		return c.Rental
	}
	return MachineLabel(c.SKU, c.GPUs)
}

// MachineLabel is how a product at a GPU count reads: `h100-sxm5-80gb`, `2x h100-sxm5-80gb`.
func MachineLabel(sku string, gpus int) string {
	if gpus > 1 {
		return fmt.Sprintf("%dx %s", gpus, sku)
	}
	return sku
}

// Chosen is the candidate the decision placed the run on.
func (d PlacementDecision) Chosen() (PlacementCandidate, bool) {
	for _, c := range d.Candidates {
		if c.Verdict == VerdictChosen {
			return c, true
		}
	}
	return PlacementCandidate{}, false
}

// Unexplained names a candidate passed over with NO verdict, and is empty when the
// record is sound. It is the invariant th-151 was filed to check: choosing a dearer or
// slower machine is allowed, but passing one over with nothing said is a chooser defect.
func (d PlacementDecision) Unexplained() string {
	for _, c := range d.Candidates {
		if c.Verdict == "" {
			return c.Name()
		}
	}
	return ""
}

// Waiting is the rental the decision waits on: a fitting one whose worker has not attached.
func (d PlacementDecision) Waiting() (PlacementCandidate, bool) {
	for _, c := range d.Candidates {
		if c.Verdict == VerdictAttaching {
			return c, true
		}
	}
	return PlacementCandidate{}, false
}

// Line is the one human sentence `cozy run` prints for the decision.
func (d PlacementDecision) Line() string {
	c, ok := d.Chosen()
	if !ok {
		if w, waiting := d.Waiting(); waiting {
			return fmt.Sprintf("placement: wait for %s to attach — %s", w.describe(), d.Tier)
		}
		return "placement: nothing chosen"
	}
	line := "placement: buy " + c.describe()
	if c.Attached() {
		line = "placement: reuse " + c.describe()
	}
	switch {
	case c.Measured:
		return fmt.Sprintf("%s — %s, %.0f s, $%.2f", line, d.Tier, c.TimeS, float64(c.CostUSDMicros)/1e6)
	case c.Rung > 0:
		return fmt.Sprintf("%s — %s, unmeasured (rung %d)", line, d.Tier, c.Rung)
	}
	return fmt.Sprintf("%s — %s, unmeasured", line, d.Tier)
}

// describe is the candidate as a line names it: `morgiana (h100-80, fp8)` for a rental,
// `h100-80 (fp8)` or `2x h100-80 (fp8)` for a product.
func (c PlacementCandidate) describe() string {
	sku := MachineLabel(c.SKU, c.GPUs)
	name, detail := sku, []string{}
	if c.Attached() {
		name, detail = c.Name(), append(detail, sku)
	}
	if c.Lane != "" {
		detail = append(detail, c.Lane)
	}
	if len(detail) == 0 {
		return name
	}
	return name + " (" + strings.Join(detail, ", ") + ")"
}

// verdicts renders every candidate with its verdict, for the daemon log.
func (d PlacementDecision) verdicts() string {
	parts := make([]string, 0, len(d.Candidates))
	for _, c := range d.Candidates {
		parts = append(parts, c.Name()+" "+c.Verdict)
	}
	return strings.Join(parts, ", ")
}

// payload is the record as the `request.placement` event carries it, plus the line.
func (d PlacementDecision) payload() map[string]any {
	raw, _ := json.Marshal(d)
	out := map[string]any{}
	_ = json.Unmarshal(raw, &out)
	out["line"] = d.Line()
	return out
}

// Exclusion reasons an attached rental can carry behind VerdictExcluded. They are
// properties of the FLEET at the moment of the decision, not of the request, so a
// reader can tell "nothing was available" from "nothing was eligible" (cl-132).
const (
	// ExcludedProtocol: new directives use the current generated wire contract.
	ExcludedProtocol = "protocol_unsupported"
	// ExcludedSpent: a completed managed job rental is retained custody, not capacity.
	ExcludedSpent = "managed_job_spent"
	// ExcludedModeConflict: the rental's worker already holds the other half of the
	// `oneof mode` — a job where a serving set is wanted, or the reverse.
	ExcludedModeConflict = "mode_conflict"
	// ExcludedWrongClass: CPU rental cannot satisfy an accelerator requirement.
	ExcludedWrongClass = "wrong_class"
	// ExcludedNotReady is spelled with the hub's own state word appended: the rental is
	// FINISHED — its acquisition failed, or it is being or has been given back — so no
	// request may wait for it. A rental merely on its way is VerdictAttaching, not this
	// (cl-185).
	ExcludedNotReady = "not_ready"
	// ExcludedAttaching is spelled with the machine appended: a fitting rental is
	// attaching, and nothing is chosen or bought past it (cl-170).
	ExcludedAttaching = "attaching"
	// ExcludedVRAMShort is spelled with the need and the device's memory appended: what
	// the pinned lanes need does not fit (cl-168, cl-170).
	ExcludedVRAMShort = "vram_short"
	// ExcludedBaseMismatch is spelled with the pod's own reason appended: the release's
	// requirements contradict the product's base image.
	ExcludedBaseMismatch = "base_mismatch"
	// ExcludedWidthUndeclared is spelled with the product's width and the degrees the
	// package declares: a machine wider than one card serves a placement as ONE group of
	// that degree, and a package whose author declared no such degree cannot be sharded
	// across it. The worker would refuse `device_group_unsupported` on arrival, so the
	// exclusion belongs here, before the pod is paid for (cl-179).
	ExcludedWidthUndeclared = "width_undeclared"
	// ExcludedDiskShort is spelled with the rental's disk and the need appended: the
	// container disk the Hub bought cannot hold the request's ingest.
	ExcludedDiskShort = "disk_short"
)

// Orchestrator is the Cozy daemon's scheduling role.
type Orchestrator struct {
	opt Options

	// done closes when the daemon is closing; Serve blocks on it. closeOnce makes Close
	// idempotent — harnesses close defensively and twice is not an event.
	done      chan struct{}
	closeOnce sync.Once
	// closingCtx ends when Close begins, so work still dialing a machine gives up.
	closingCtx    context.Context
	cancelClosing context.CancelFunc

	mu    sync.Mutex
	waits map[string]*wait // by request#attempt
	// outputExporting serializes retries of one durable local --out obligation.
	outputExporting map[string]bool
	// pending is the queue of machine executions owed a start, in submission order.
	pending []string
	// parked is the last wait each queued request announced, by id. An entry lives
	// exactly as long as its request is in `pending`.
	parked map[string]string
	// closing is set by Close: nothing new starts on a daemon that is going away.
	closing bool
	// starting names the machine executions whose start is in flight.
	starting map[string]bool
	events   []string

	// phases is the preparation-phase lane (phase.go): what a request is doing before
	// its first attempt exists. Live-only and observational.
	phases *phases
	// transferWake is a lossy nudge over durable request-attached transfer rows.
	transferWake    map[string]chan struct{}
	transferRunning map[string]bool
	transferWork    map[string]*transferWork
}

type wait struct {
	accepted chan struct{}
	closed   chan struct{}
	once     sync.Once
	onceA    sync.Once
	refs     int
	err      *exit.Error
}

// Open builds the orchestrator. No listener binds here: the owner DIALS each worker's
// own socket (#436); a second Cozy daemon on one root fails on the daemon lock instead.
func Open(opt Options) (*Orchestrator, *exit.Error) {
	if opt.Log == nil {
		opt.Log = io.Discard
	}
	c := &Orchestrator{
		opt:             opt,
		done:            make(chan struct{}),
		waits:           map[string]*wait{},
		outputExporting: map[string]bool{},
		starting:        map[string]bool{},
		parked:          map[string]string{},
		phases:          newPhases(),
		transferWake:    make(map[string]chan struct{}),
		transferRunning: make(map[string]bool),
		transferWork:    make(map[string]*transferWork),
	}
	c.closingCtx, c.cancelClosing = context.WithCancel(context.Background())
	return c, nil
}

// Serve blocks until Close. The owner dials workers; there is no server to run (#436),
// and the blocking shape is kept so entrypoints stay one-line callers.
func (c *Orchestrator) Serve() error {
	<-c.done
	return nil
}

// Close ends the daemon's own work: machines keep running theirs.
func (c *Orchestrator) Close(grace time.Duration) {
	c.mu.Lock()
	c.closing = true
	c.mu.Unlock()
	c.cancelClosing()
	c.closeOnce.Do(func() { close(c.done) })
}

func (c *Orchestrator) logf(format string, args ...any) {
	line := fmt.Sprintf("[orchestrator %s] ", time.Now().UTC().Format("15:04:05.000")) +
		fmt.Sprintf(format, args...)
	fmt.Fprintln(c.opt.Log, line)
	c.mu.Lock()
	c.events = append(c.events, line)
	if len(c.events) > 512 {
		c.events = c.events[len(c.events)-512:]
	}
	c.mu.Unlock()
}

// Events is the orchestrator's own recent activity, for status rendering. Runtime events
// reach a client through here and never bypass the orchestrator.
func (c *Orchestrator) Events() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]string(nil), c.events...)
}

// Store is the lifecycle authority this orchestrator writes. cl-006's API reads requests,
// attempts, outputs and durable events through it — the same rows, never a second copy.
func (c *Orchestrator) Store() *records.Store { return c.opt.Store }

// Layout is the local root this orchestrator grants into.
func (c *Orchestrator) Layout() home.Layout { return c.opt.Layout }

// emit appends ONE durable lifecycle event. A failure to append is logged and never
// fatal: the authority's own row is the fact, and the stream is its announcement.
func (c *Orchestrator) emit(requestID, eventType string, attempt uint64, payload map[string]any) {
	if e := c.opt.Store.AppendEvent(requestID, eventType, int64(attempt), payload); e != nil {
		c.logf("event %s for %s NOT appended: %s", eventType, requestID, e.Message)
	}
}

func key(requestID string, attempt uint64) string {
	return fmt.Sprintf("%s#%d", requestID, attempt)
}

// acquireWait retains only live callers. Terminal/acceptance signals never create a map
// entry of their own, so fire-and-forget requests cannot accumulate one wait forever.
func (c *Orchestrator) acquireWait(k string) (*wait, func()) {
	c.mu.Lock()
	w, ok := c.waits[k]
	if !ok {
		w = &wait{accepted: make(chan struct{}), closed: make(chan struct{})}
		c.waits[k] = w
	}
	w.refs++
	c.mu.Unlock()
	return w, func() {
		c.mu.Lock()
		w.refs--
		if w.refs == 0 && c.waits[k] == w {
			delete(c.waits, k)
		}
		c.mu.Unlock()
	}
}

func (c *Orchestrator) signalClosed(k string, e *exit.Error) {
	c.mu.Lock()
	w := c.waits[k]
	c.mu.Unlock()
	if w != nil {
		w.markClosed(e)
	}
}

// enqueue parks a request until some worker advertises its binding as ready.
func (c *Orchestrator) enqueue(requestID string) bool {
	if link, problem := c.opt.Store.MachineExecution(requestID); problem != nil || link != nil {
		return false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	// Cancellation and asynchronous activation race at this boundary. Read the
	// durable state while holding the same lock CancelQueued uses to remove an id:
	// either activation appends first and cancel removes it, or activation observes
	// the absorbing terminal and appends nothing.
	row, problem := c.opt.Store.RequestRow(requestID)
	if problem != nil || row == nil || (row.State != "submitted" && row.State != "queued") {
		return false
	}
	for _, id := range c.pending {
		if id == requestID {
			return true
		}
	}
	c.pending = append(c.pending, requestID)
	return true
}

// QueuePosition is where a waiting request sits in the dispatch queue, counted from 1.
// Zero means it is not waiting — either it never queued or it already has an attempt.
// The queue is a slice appended in submission order and drained in that order, so the
// position is the real thing rather than an estimate (cr-019: many queued jobs against
// one worker drain FIFO, and a client can watch it happen).
func (c *Orchestrator) QueuePosition(requestID string) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	for i, id := range c.pending {
		if id == requestID {
			return i + 1
		}
	}
	return 0
}

// QueueState describes the request's machine queue. The dispatcher visits one global
// list, but requests pinned to another machine do not occupy this machine's queue.
// Read both numbers from the same pending snapshot and exclude settled/open attempts.
func (c *Orchestrator) QueueState(requestID string) (position, depth int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	owed, problem := c.opt.Store.Owed()
	if problem != nil {
		return 0, 0
	}
	venues := make(map[string]string, len(owed))
	for _, request := range owed {
		venue := request.Worker
		if venue == "" {
			venue = request.RequestedRental
		}
		if venue == "" && request.RentalRequired {
			venue = "unassigned-rental"
		}
		venues[request.ID] = venue
	}
	venue, found := venues[requestID]
	if !found {
		return 0, 0
	}
	for _, id := range c.pending {
		if other, waiting := venues[id]; !waiting || other != venue {
			continue
		}
		depth++
		if id == requestID {
			position = depth
		}
	}
	return position, depth
}

// reviveQueue re-asks the queue's residency question after the capacity it depends on
// changed: a worker's process went, a launch finished, a request left. It asks
// select-or-start for the HEAD per machine: reviving every waiting
// request would let two requests needing different plans on one card stop each other's
// worker in turn. Nothing is dispatched here — `drain` is still the one placement path.
func (c *Orchestrator) reviveQueue() {
	c.mu.Lock()
	closing, queued := c.closing, append([]string(nil), c.pending...)
	c.mu.Unlock()
	if closing {
		return
	}
	asked := map[string]bool{}
	for _, id := range queued {
		req, e := c.opt.Store.RequestRow(id)
		if e != nil || req == nil {
			continue
		}
		machine := req.Worker
		if machine == "" {
			machine = req.RequestedRental
		}
		if machine == "" && req.Rental {
			// The fleet places an unassigned --rental request, not a local card: only
			// requests for the same placement slot wait on one another.
			machine = "rental/" + requestSlot(*req)
		}
		if asked[machine] {
			continue
		}
		asked[machine] = true
		c.start(*req)
	}
}

// WakeQueue re-asks durable queued work after an external capacity observation changes. It is
// intentionally edge-triggered by the fleet observer; the queue remains the authority and
// duplicate wakes cannot mint duplicate attempts.
func (c *Orchestrator) WakeQueue() {
	c.reviveQueue()
}

// CancelQueued settles a request that is WAITING and has no attempt to cancel. It leaves
// the queue and is settled canceled — a client that asked for a cancel is owed an answer,
// and "it will start later anyway" is not one. `actor` is who asked (cl-108): every
// cancellation is attributed in its own durable terminal, never an anonymous verdict.
func (c *Orchestrator) CancelQueued(requestID, actor string) *exit.Error {
	if actor == "" {
		actor = "an unnamed client"
	}
	payload := map[string]any{
		"status": "CANCELED", "cause": "CLIENT_CANCELED",
		"error_type": "CLIENT_CANCELED", "actor": actor,
		"error": fmt.Sprintf(
			"canceled by %s before any attempt was dispatched", actor),
		"outputs": []any{}, "requeuing": false,
	}
	applied, e := c.opt.Store.CancelQueuedRequest(requestID, payload)
	if e != nil || !applied {
		return e
	}
	// CancelQueuedRequest committed the request terminal and its event together. Settle
	// the independent --out obligation too: attempt zero can never produce publishable
	// bytes, and `pending` must not outlive an absorbing request terminal.
	c.RetryOutputExport(requestID)
	row, problem := c.opt.Store.RequestRow(requestID)
	if problem != nil || row == nil {
		if problem != nil {
			return problem
		}
		return exit.Internalf("canceled request %s cannot be read back", requestID)
	}
	// The canceled terminal is only honest once nothing moves this request's bytes.
	c.stopTransfer(requestID)
	if row.ModelTransfer != nil {
		c.forgetTransferProgress(requestID)
		c.signalTransfer(requestID)
	}
	c.forget(requestID)
	c.logf("%s canceled before any attempt", requestID)
	c.signalClosed(requestWaitKey(requestID),
		exit.New(exit.Canceled, "%s was canceled before any attempt was dispatched", requestID))
	c.cleanupRequestAssets(*row)
	c.reviveQueue()
	return nil
}

func (c *Orchestrator) forget(requestID string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := c.pending[:0]
	for _, id := range c.pending {
		if id != requestID {
			out = append(out, id)
		}
	}
	c.pending = out
	delete(c.parked, requestID)
}

// requestWaitKey names the request-level wait, which may span several attempt ordinals.
func requestWaitKey(requestID string) string { return "request:" + requestID }

func (w *wait) markAccepted() { w.onceA.Do(func() { close(w.accepted) }) }
func (w *wait) markClosed(e *exit.Error) {
	w.once.Do(func() {
		w.err = e
		close(w.closed)
	})
}
