package orchestrator

// Preparation observations explain queued work before its execution attempt exists.
// Producers supply phases and, when measured, byte counters. Rates and estimates
// are derived from observed progress, excluding bytes retained before observation.
// This is live display state; it never settles, routes, bills, or cancels work.

import (
	"sort"
	"strings"
	"sync"
	"time"
)

// The phase vocabulary. Ordered as a request traverses it, though nothing depends on the
// order: each name is reported by the party that observed it, never inferred from the
// one before.
const (
	// PhaseAcquiring: a machine is being chosen or bought, and the provider has not yet
	// confirmed a container exists.
	PhaseAcquiring = "acquiring"
	// PhaseReplanning: an acquisition FAILED and the rental is buying again. It is its
	// own phase because it is the one thing in this sequence that spends money and
	// produces nothing, and because the request may quietly change datacenter doing it —
	// which changes whether a repo cache is there at all. Measured on run 207: 32.1s of a
	// 227.9s wait was one datacenter declining to create a pod, and an operator watching
	// `queued` had no way to learn either that it happened or that the request moved.
	PhaseReplanning = "replanning"
	// PhaseProvisioning: the provider holds the request and has not reported the
	// container running. This is the provider's queue, and it is not ours to shorten.
	PhaseProvisioning = "provisioning"
	// PhasePullingImage: the provider explicitly reports a container image pull.
	PhasePullingImage = "pulling_image"
	// PhaseBooting: the paid pod exists and has not yet reported itself attachable.
	PhaseBooting = "booting"
	// PhaseResolving: the pod host verified the request and journaled the intent.
	PhaseResolving = "resolving"
	// PhaseDownloading: model bytes are landing in the pod's own store.
	PhaseDownloading = "downloading"
	// PhasePreparing: verified local files were handed to Runtime preparation.
	PhasePreparing = "preparing"
	// PhaseWarming: the prepared placement is converging toward dispatchable.
	PhaseWarming = "warming"
	// PhaseGPUWait: Runtime accepted the execution and holds its call until the GPUs it
	// needs are free (its gpu.wait event names the width and the roots ahead of it).
	PhaseGPUWait = "gpu_wait"
	// PhaseOwnerReconciliation: a sent publication's machine authorization expired, so the
	// owner's own Hub read settles it (PhaseDetail names the publication).
	PhaseOwnerReconciliation = "owner_reconciliation"
)

// PreparationRateMaxAge bounds the freshness of displayed transfer estimates only.
// It never cancels, fails, or otherwise changes execution.
const PreparationRateMaxAge = 10 * time.Second

// PhaseObservation is one subject's current phase as last observed.
//
// Rate and Average are both bytes per second and they are different measurements on
// purpose. Average covers progress since the first byte sample and is what a remaining-time
// estimate divides by; Rate is the most recent interval alone and is therefore what shows
// a link degrading while it degrades. Reporting one number for both jobs would make it
// wrong for one of them.
type PhaseObservation struct {
	Name       string
	Since      time.Time
	At         time.Time
	Machine    string
	Detail     string
	Rental     *RentalProgress
	Models     []ModelDownloadProgress
	WaitingFor *WaitingRun
	// HasBytes is false for a phase whose producer declared no counters. Moved and Total
	// are then meaningless and must not be rendered.
	HasBytes bool
	Moved    uint64
	// Total is 0 when the producer declared no denominator. A fraction is rendered only
	// when it is nonzero; one is never assembled from anything else.
	Total   uint64
	Rate    float64
	Average float64
}

// RentalProgress is the observed rental product and its whole-pod hourly rate.
// It is display information only; billing and admission keep their own authority.
type RentalProgress struct {
	AcceleratorModel      string `json:"accelerator_model"`
	AcceleratorCount      int    `json:"accelerator_count"`
	HourlyRateUSDMicros   int64  `json:"hourly_rate_usd_micros"`
	BaseWorkerImageDigest string `json:"base_worker_image_digest,omitempty"`
	BaseWorkerImageTag    string `json:"base_worker_image_tag,omitempty"`
	BaseWorkerProfile     string `json:"base_worker_profile,omitempty"`
}

// Elapsed is how long this phase has been the current one.
func (p PhaseObservation) Elapsed() time.Duration {
	if p.Since.IsZero() {
		return 0
	}
	return time.Since(p.Since)
}

// SampleAge is the age of the last actual producer observation.
func (p PhaseObservation) SampleAge() time.Duration {
	if p.At.IsZero() {
		return 0
	}
	return max(time.Since(p.At), 0)
}

// Remaining is the measured estimate: bytes still owed divided by the rate this phase has
// actually sustained. False whenever either quantity is missing, which is every phase
// with no denominator and every phase before two samples have landed.
func (p PhaseObservation) Remaining() (time.Duration, bool) {
	if !p.HasBytes || p.Total == 0 || p.Moved >= p.Total || p.Average <= 0 || p.Rate <= 0 || p.SampleAge() > PreparationRateMaxAge {
		return 0, false
	}
	seconds := float64(p.Total-p.Moved) / p.Average
	if seconds <= 0 || seconds > float64(time.Duration(1<<62)/time.Second) {
		return 0, false
	}
	return time.Duration(seconds * float64(time.Second)), true
}

// phaseState is one subject's accumulator. `prev` is the immediately preceding sample and
// exists only to measure the most recent interval.
type phaseState struct {
	name        string
	since       time.Time
	at          time.Time
	machine     string
	detail      string
	rental      *RentalProgress
	models      map[string]modelDownloadState
	hasBytes    bool
	moved       uint64
	total       uint64
	byteSince   time.Time
	initialMove uint64
	prevAt      time.Time
	prevMove    uint64
	rate        float64
}

// phases holds one observation per subject. A subject is a request id (the phases a
// request traverses before any worker is bound to it) or a worker instance id (the phases
// a preparation traverses on that worker, which every request waiting on it shares).
type phases struct {
	mu       sync.Mutex
	subjects map[string]*phaseState
}

func newPhases() *phases { return &phases{subjects: map[string]*phaseState{}} }

// PhaseSample is one report. Producers fill only what they measured: a producer with no
// counters leaves HasBytes false, and a producer with no denominator leaves Total zero.
type PhaseSample struct {
	Name     string
	Machine  string
	Detail   string
	Rental   *RentalProgress
	Models   []ModelDownloadProgress
	HasBytes bool
	Moved    uint64
	Total    uint64
}

// observe folds one sample into the subject's accumulator.
//
// A NAME CHANGE RESTARTS THE MEASUREMENT. Rate is a fact about the phase being measured,
// so carrying a download's rate into the preparation that follows it would report a number
// about work that is over.
//
// Counters are monotonic within one call. A worker can begin another fetch with
// smaller counters; that resets the rate basis without resetting phase elapsed time.
func (p *phases) observe(subject string, sample PhaseSample) {
	if subject == "" || sample.Name == "" {
		return
	}
	now := time.Now()
	p.mu.Lock()
	defer p.mu.Unlock()
	state := p.subjects[subject]
	if state == nil || state.name != sample.Name {
		state = &phaseState{name: sample.Name, since: now}
		p.subjects[subject] = state
	}
	if sample.Machine != "" {
		state.machine = sample.Machine
	}
	state.detail = sample.Detail
	if sample.Rental != nil {
		rental := *sample.Rental
		state.rental = &rental
	}
	if sample.HasBytes {
		moved := sample.Moved
		if sample.Total > 0 {
			moved = min(moved, sample.Total)
		}
		if !state.hasBytes || moved < state.prevMove {
			// One worker can report a new fetch while retaining the same phase.
			// Its initial counter is a new baseline, not newly received bytes.
			state.byteSince, state.initialMove, state.rate = now, moved, 0
		} else if now.After(state.prevAt) {
			state.rate = float64(moved-state.prevMove) / now.Sub(state.prevAt).Seconds()
		}
		state.prevAt, state.prevMove = now, moved
		state.hasBytes, state.moved, state.total = true, moved, sample.Total
	}
	state.observeModels(sample.Models, now)
	state.at = now
}

func (p *phases) snapshot(subject string) (PhaseObservation, bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	state := p.subjects[subject]
	if state == nil {
		return PhaseObservation{}, false
	}
	out := PhaseObservation{
		Name: state.name, Since: state.since, At: state.at, Machine: state.machine,
		Detail: state.detail, HasBytes: state.hasBytes, Moved: state.moved,
		Total: state.total, Rate: state.rate,
	}
	if state.rental != nil {
		rental := *state.rental
		out.Rental = &rental
	}
	out.Models = state.modelSnapshots(time.Now())
	if state.hasBytes {
		out.At = state.prevAt // byte-less heartbeats cannot freshen old byte estimates
		if elapsed := state.prevAt.Sub(state.byteSince).Seconds(); elapsed > 0 && state.moved > state.initialMove {
			out.Average = float64(state.moved-state.initialMove) / elapsed
		}
	}
	if out.SampleAge() > PreparationRateMaxAge {
		out.Rate, out.Average = 0, 0
	}
	return out, true
}

func (p *phases) forget(subject string) {
	if subject == "" {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	delete(p.subjects, subject)
}

// ObservePhase records one phase sample against a subject. It is the ONE entry point:
// rental acquisition, the pod host's prepare stream, and the placement convergence all
// report through it, so there is one accumulator and one set of rules about what a
// missing number means.
//
// It also FANS THE SAMPLE OUT to whoever is attached to a request the subject is
// preparing for, over the lossy live lane. Without that, an attached `cozy run` learns
// nothing until the next durable queue event, and during a materialization there is no
// next durable queue event: the drain runs on worker reports, and a worker mid-download
// reports nothing for hours. That is precisely how a healthy run became indistinguishable
// from a hung one.
func (c *Orchestrator) ObservePhase(subject string, sample PhaseSample) {
	c.phases.observe(subject, sample)
	for _, requestID := range c.phaseAudience(subject) {
		// A shared worker observation must not replace this request's FIFO or
		// capacity wait. Live frames and newly attached watchers use one view.
		if observed, ok := c.QueuePhase(requestID); ok {
			c.frames.publish(observed.Frame(requestID))
		}
	}
}

// phaseAudience names the queued requests this subject's phase describes. A request-scoped
// subject describes exactly itself; a worker-scoped one may affect queued requests
// waiting on that worker. Each recipient still projects its own current queue wait
// before the observation is broadcast; it does not inherit raw worker preparation.
func (c *Orchestrator) phaseAudience(subject string) []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	audience := make([]string, 0, len(c.pending))
	for _, requestID := range c.pending {
		if requestID == subject || c.preparingFor(requestID) == subject {
			audience = append(audience, requestID)
		}
	}
	return audience
}

// Frame uses the same phase document for live updates and initial SSE snapshots.
func (p PhaseObservation) Frame(requestID string) Frame {
	return Frame{RequestID: requestID, Type: "phase", Value: p.wire()}
}

// wire is the live frame's payload. Absent keys mean "not measured": a consumer must not
// read a missing rate as zero, so a phase with no counters carries no counter keys at all.
func (p PhaseObservation) wire() map[string]any {
	out := map[string]any{"phase": p.Name}
	if !p.At.IsZero() {
		out["sample_age_ms"] = p.SampleAge().Milliseconds()
	}
	if p.Machine != "" {
		out["machine"] = p.Machine
	}
	if p.Detail != "" {
		out["detail"] = p.Detail
	}
	if p.Rental != nil {
		out["rental"] = p.Rental
	}
	if len(p.Models) > 0 {
		out["models"] = p.Models
	}
	if p.WaitingFor != nil {
		out["waiting_for"] = p.WaitingFor
	}
	if elapsed := p.Elapsed(); elapsed > 0 {
		out["elapsed_ms"] = elapsed.Milliseconds()
	}
	if p.HasBytes {
		out["moved_bytes"] = p.Moved
		if p.Total > 0 {
			out["total_bytes"] = p.Total
		}
		if p.Rate > 0 && p.SampleAge() <= PreparationRateMaxAge {
			out["rate_bytes_per_second"] = p.Rate
		}
		if remaining, ok := p.Remaining(); ok {
			out["remaining_ms"] = remaining.Milliseconds()
		}
	}
	return out
}

// QueuePhase answers what a request that has not yet dispatched is doing, for the surfaces
// that render it. A current capacity wait wins over old preparation observations;
// otherwise a real preparation observation wins and the routing's wait cause stands in
// so the column is never blank while the request is genuinely waiting on capacity. The
// stand-in carries no timing, because a wait cause is a classification and not a
// measurement — and inventing an elapsed for it would be exactly the fabrication this lane
// refuses everywhere else.
func (c *Orchestrator) QueuePhase(requestID string) (PhaseObservation, bool) {
	row, problem := c.opt.Store.RequestRow(requestID)
	if problem != nil || row == nil || (row.State != "submitted" && row.State != "queued") {
		return PhaseObservation{}, false
	}
	facts := c.waitOf(*row)
	if facts.cause != WaitSlotBusy && facts.cause != WaitQueueAhead {
		if observed, ok := c.PhaseOf(requestID); ok {
			return c.phaseRental(requestID, observed), true
		}
	}
	if facts.cause == "" {
		return PhaseObservation{}, false
	}
	return c.phaseRental(requestID, PhaseObservation{Name: facts.cause, Machine: facts.on,
		WaitingFor: facts.waitingFor}), true
}

// PreparationPhase is one subject's observation read directly — the worker instance a
// preparation runs on, or a request in acquisition. `PhaseOf` is the request-shaped
// question; this is the subject-shaped one, for callers that already hold the subject.
func (c *Orchestrator) PreparationPhase(subject string) (PhaseObservation, bool) {
	return c.phases.snapshot(subject)
}

// ForgetPhase drops a subject's observation. Called when the subject stops existing: a
// request settles, a worker exits, a preparation ends.
func (c *Orchestrator) ForgetPhase(subject string) { c.phases.forget(subject) }

// PhaseOf answers what a not-yet-dispatched request is actually doing, or false when
// nothing has been observed about it.
//
// Two subjects can hold an answer and the order between them is the point: a phase
// recorded against the REQUEST is about getting it a machine, and one recorded against
// the WORKER is about making that machine serve it. The request's own phase wins while it
// exists, because a request that has not been bound to a worker cannot be described by
// any worker's phase.
func (c *Orchestrator) PhaseOf(requestID string) (PhaseObservation, bool) {
	if requestID == "" {
		return PhaseObservation{}, false
	}
	if observed, ok := c.phases.snapshot(requestID); ok {
		return observed, true
	}
	c.mu.Lock()
	instance := c.preparingFor(requestID)
	c.mu.Unlock()
	if instance == "" {
		return PhaseObservation{}, false
	}
	return c.phases.snapshot(instance)
}

// preparingFor names the worker whose preparation this queued request is waiting on. It
// is the same question `classifyCapacityWait` asks and is answered the same way — the
// worker pinned by the request's rental, or the live worker in the request's own slot —
// so the phase shown can never belong to a machine the request is not waiting for.
// Callers hold c.mu.
func (c *Orchestrator) preparingFor(requestID string) string {
	row, problem := c.opt.Store.RequestRow(requestID)
	if problem != nil || row == nil {
		return ""
	}
	slot := pinnedPackage(row.Package, row.Worker)
	for _, w := range c.workers {
		if w.exited || w.stopping {
			continue
		}
		if row.Worker != "" && w.spec.Connection != nil {
			if w.instanceID == rentalInstanceID(row.Worker) {
				return w.instanceID
			}
			continue
		}
		if w.spec.Connection == nil && w.spec.Placement.Package == slot {
			return w.instanceID
		}
	}
	return ""
}

// PhaseOfHubRental maps what the hub says about a rental onto this vocabulary, and only
// what it says. The empty answer means "this rental is not in a preparation phase" — a
// ready pod has left acquisition, and a failed one is a terminal the request settles on.
//
// THE BOUNDARY THAT MATTERS IS `provisioning` -> `booting`, and it is drawn where the
// PROVIDER draws it. Before the provider reports the container running, the pod is waiting
// in someone else's queue and nothing we ship can change how long that takes. After it,
// the time is the container image coming down and the supervisor coming up, both of which
// are ours. Collapsing the two — which is what one `pending_acquisition` edge does — makes
// a 9 GB image pull and a provider stock-out look identical, and they have opposite
// remedies.
//
// A hub that reports no provider words yields `acquiring` and stops there. That is not a
// degraded rendering to apologize for: it is the honest one. Splitting it further would
// mean deciding a phase from elapsed time, which is the thing this whole lane exists to
// avoid.
func PhaseOfHubRental(state, providerState, containerState string, retrying bool) string {
	if strings.TrimSpace(state) != "pending_acquisition" {
		return ""
	}
	// A rental back in pending_acquisition CARRYING A FAILURE is on its second (or later)
	// ordinal: the first one was bought, paid for, and refused. That is not the same
	// wait as the first attempt and must not render as one.
	if retrying {
		return PhaseReplanning
	}
	if strings.TrimSpace(providerState) == "" {
		// Nothing bought yet. This is the hub's own placement and selection, and it is
		// NOT a rounding error: four consecutive runs on one SKU spent between 4.6s and
		// 25.3s here, before the provider was asked at all. Naming it is the only way
		// anyone learns it exists.
		return PhaseAcquiring
	}
	switch strings.ToUpper(strings.TrimSpace(containerState)) {
	case "PULLING", "PULLING_IMAGE", "IMAGE_PULLING":
		return PhasePullingImage
	}
	if providerRunning(providerState, containerState) {
		return PhaseBooting
	}
	return PhaseProvisioning
}

// providerRunning reads the provider's own two words for "this container is executing".
// The comparison is case-insensitive and covers the spellings providers actually use;
// anything else is read as not-yet-running, which keeps an unrecognised word on the
// pessimistic side of the boundary rather than announcing a boot that has not started.
func providerRunning(providerState, containerState string) bool {
	running := func(word string) bool {
		switch strings.ToUpper(strings.TrimSpace(word)) {
		case "RUNNING", "RUNNING_HEALTHY", "EXITED", "TERMINATED":
			return true
		}
		return false
	}
	if strings.TrimSpace(containerState) != "" {
		return running(containerState)
	}
	return running(providerState)
}

// phaseRental fills the display from the existing rental row when preparation has
// moved from the request to its worker. It introduces no second persisted quote.
func (c *Orchestrator) phaseRental(requestID string, phase PhaseObservation) PhaseObservation {
	if phase.Rental != nil {
		return phase
	}
	request, problem := c.opt.Store.RequestRow(requestID)
	if problem != nil || request == nil || request.Worker == "" {
		return phase
	}
	rental, problem := c.opt.Store.RentalByMachine(request.Worker)
	if problem != nil || rental == nil {
		return phase
	}
	if phase.Machine == "" {
		phase.Machine = rental.MachineName
	}
	phase.Rental = &RentalProgress{AcceleratorModel: rental.AcceleratorModel,
		AcceleratorCount: rental.AcceleratorCount, HourlyRateUSDMicros: rental.HourlyRateUSDMicros}
	return phase
}

// ModelDownloadProgress names one checkpoint and measures its available bytes.
// Rate excludes bytes already present at the start of the fetch. A cache copy is
// transferred work, but is distinct from network origin bytes.
type ModelDownloadProgress struct {
	Model       string  `json:"model"`
	Release     string  `json:"release"`
	Lane        string  `json:"lane"`
	Slot        string  `json:"slot,omitempty"`
	Manifest    string  `json:"manifest"`
	Moved       uint64  `json:"moved_bytes"`
	Total       uint64  `json:"total_bytes,omitempty"`
	OriginBytes uint64  `json:"origin_bytes,omitempty"`
	CachedBytes uint64  `json:"cached_bytes,omitempty"`
	Rate        float64 `json:"rate_bytes_per_second,omitempty"`
	RemainingMS *int64  `json:"remaining_ms,omitempty"`
	SampleAgeMS int64   `json:"sample_age_ms"`
}

type modelDownloadState struct {
	value             ModelDownloadProgress
	since, previousAt time.Time
	initial, previous uint64
}

func (p *phaseState) observeModels(samples []ModelDownloadProgress, now time.Time) {
	if len(samples) == 0 {
		return
	}
	if p.models == nil {
		p.models = map[string]modelDownloadState{}
	}
	for _, sample := range samples {
		if sample.Model == "" || sample.Manifest == "" {
			continue
		}
		sample.Rate, sample.RemainingMS = 0, nil
		key := sample.Model + "\x00" + sample.Release + "\x00" + sample.Lane + "\x00" + sample.Manifest
		moved := sample.OriginBytes + sample.CachedBytes
		old, exists := p.models[key]
		if !exists || moved < old.previous || sample.Moved < old.value.Moved {
			old = modelDownloadState{since: now, previousAt: now, initial: moved, previous: moved}
		}
		if now.After(old.previousAt) {
			sample.Rate = float64(moved-old.previous) / now.Sub(old.previousAt).Seconds()
		}
		old.previousAt, old.previous = now, moved
		if elapsed := now.Sub(old.since).Seconds(); elapsed > 0 && sample.Rate > 0 && moved > old.initial && sample.Total > sample.Moved {
			average := float64(moved-old.initial) / elapsed
			seconds := float64(sample.Total-sample.Moved) / average
			if seconds > 0 && seconds < float64(time.Duration(1<<62)/time.Second) {
				remaining := int64(seconds * 1000)
				sample.RemainingMS = &remaining
			}
		}
		old.value = sample
		p.models[key] = old
	}
}

func (p *phaseState) modelSnapshots(now time.Time) []ModelDownloadProgress {
	keys := make([]string, 0, len(p.models))
	for key := range p.models {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	out := make([]ModelDownloadProgress, 0, len(keys))
	for _, key := range keys {
		state := p.models[key]
		value := state.value
		value.SampleAgeMS = max(now.Sub(state.previousAt), 0).Milliseconds()
		if now.Sub(state.previousAt) > PreparationRateMaxAge {
			value.Rate, value.RemainingMS = 0, nil
		}
		out = append(out, value)
	}
	return out
}
