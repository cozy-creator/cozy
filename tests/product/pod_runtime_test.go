package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// podRuntime plays the minor-61 Runtime (proto-061 C/D1/D2) on the fake pod's control
// stream: one placement per package release whose id and model ids are stable as slots are
// added, every construction the set binds DISPATCHABLE at once and still so after the set
// grows (no drain), an attempt that runs until the test finishes it, and
// `loaded_binding_digests` naming each construction an attempt has entered.
type podRuntime struct {
	// hold keeps a desired state recorded but unaccepted, as a Runtime still staging it
	// reports the set it already holds.
	hold func(*pb.DesiredWorkerState) bool
	// lanes reports proto-061 G's lanes: every placement on lane-0, which admits the pod's
	// slots before a device release — the head RUNNING, the rest QUEUED behind it — and an
	// idle lane-1 with as many, so the worker-level sum over-advertises lane-0.
	lanes bool

	mu       sync.Mutex
	order    []string // running, in admission order
	accepted *pb.DesiredWorkerState
	loaded   map[string]bool
	running  map[string]*pb.AttemptOffer
	send     func(*pb.WorkerFrame) error
}

func newPodRuntime() *podRuntime {
	return &podRuntime{loaded: map[string]bool{}, running: map[string]*pb.AttemptOffer{}}
}

func (r *podRuntime) frame(p *fakePod, frame *pb.RecordOwnerFrame, send func(*pb.WorkerFrame) error) (bool, error) {
	switch m := frame.Msg.(type) {
	case *pb.RecordOwnerFrame_DesiredState:
		d := m.DesiredState
		if d.GetPlacementSet() == nil {
			return false, nil
		}
		p.mu.Lock()
		p.desired = append(p.desired, d)
		p.lanes = append(p.lanes, "placement_set")
		p.mu.Unlock()
		r.mu.Lock()
		r.send = send
		if r.hold == nil || !r.hold(d) {
			r.accepted = d
		}
		r.mu.Unlock()
		return true, r.report(p)
	case *pb.RecordOwnerFrame_AttemptOffer:
		offer := m.AttemptOffer
		spec, err := canonical.Read(offer.InvocationSpecCanonicalBytes, &pb.InvocationSpec{})
		if err != nil {
			return true, err
		}
		r.mu.Lock()
		r.running[offer.RequestId] = offer
		r.order = append(r.order, offer.RequestId)
		r.loaded[spec.Sub("serving").Str("entrypoint_binding_digest")] = true
		r.mu.Unlock()
		// Counted only once it runs: a test that waits for offer n may finish it at once.
		p.mu.Lock()
		p.offers = append(p.offers, offer)
		p.mu.Unlock()
		if err := send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_AttemptAccepted{AttemptAccepted: &pb.AttemptAccepted{
			RecordOwnerEpoch: offer.RecordOwnerEpoch, ControlStreamEpoch: offer.ControlStreamEpoch,
			WorkerBootId: offer.WorkerBootId, RequestId: offer.RequestId, AttemptOrdinal: offer.AttemptOrdinal,
			InvocationSpecDigest: offer.InvocationSpecDigest, PlacementId: offer.PlacementId,
		}}}); err != nil {
			return true, err
		}
		return true, r.report(p)
	case *pb.RecordOwnerFrame_OutcomeAck:
		return true, r.report(p)
	}
	return false, nil
}

// report is the Runtime's ObservedWorkerState for the set it accepted.
func (r *podRuntime) report(p *fakePod) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.accepted == nil || r.send == nil {
		return nil
	}
	frame := p.served(r.accepted, 1)
	if frame == nil {
		return nil
	}
	state := frame.GetObservedState()
	p.mu.Lock()
	slots := max(p.slots, 1)
	p.mu.Unlock()
	state.AvailableAttemptSlots = uint32(max(int(slots)-len(r.running), 0))
	for _, placement := range state.Placements {
		for _, digest := range placement.DispatchableBindingDigests {
			if spelled, err := canonical.Spell(digest); err == nil && r.loaded[spelled] {
				placement.LoadedBindingDigests = append(placement.LoadedBindingDigests, digest)
			}
		}
	}
	ids := make([]string, 0, len(r.running))
	for id := range r.running {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	if r.lanes {
		ids = r.order
		lane0 := &pb.DeviceLane{LaneId: "lane-0", DeviceOrdinals: []uint32{0},
			AvailableAttemptSlots: state.AvailableAttemptSlots}
		for _, placement := range state.Placements {
			placement.DeviceLaneId = "lane-0"
			lane0.PlacementIds = append(lane0.PlacementIds, placement.PlacementId)
		}
		state.Lanes = []*pb.DeviceLane{lane0,
			{LaneId: "lane-1", DeviceOrdinals: []uint32{1}, AvailableAttemptSlots: slots}}
		state.AvailableAttemptSlots += slots
	}
	for i, id := range ids {
		offer := r.running[id]
		row := &pb.HeldAttempt{RequestId: offer.RequestId,
			AttemptOrdinal: offer.AttemptOrdinal, Kind: pb.AttemptKind_ATTEMPT_KIND_SERVING,
			State: pb.AttemptState_ATTEMPT_STATE_RUNNING, InvocationSpecDigest: offer.InvocationSpecDigest,
			PlacementId: offer.PlacementId, ExecutorEpoch: 1}
		if r.lanes {
			row.LaneId = "lane-0"
			if i > 0 {
				row.State, row.ExecutorEpoch = pb.AttemptState_ATTEMPT_STATE_QUEUED, 0
				row.QueuePosition = uint32(i - 1)
			}
		}
		state.HeldAttempts = append(state.HeldAttempts, row)
	}
	return r.send(frame)
}

// finish ends a running attempt as SUCCEEDED and reports the freed seat.
func (r *podRuntime) finish(t *testing.T, p *fakePod, requestID string) {
	t.Helper()
	r.mu.Lock()
	offer := r.running[requestID]
	delete(r.running, requestID)
	for i, id := range r.order {
		if id == requestID {
			r.order = append(r.order[:i:i], r.order[i+1:]...)
			break
		}
	}
	send := r.send
	r.mu.Unlock()
	if offer == nil {
		running := make([]string, 0, len(r.running))
		r.mu.Lock()
		for id := range r.running {
			running = append(running, id)
		}
		r.mu.Unlock()
		t.Fatalf("%s is not running; running %v", requestID, running)
	}
	spelled, err := canonical.Spell(offer.InvocationSpecDigest)
	must(t, err)
	body, digest, err := canonical.Identity(&pb.AttemptOutcomeBody{RequestId: offer.RequestId,
		AttemptOrdinal: offer.AttemptOrdinal, InvocationSpecDigest: spelled,
		Status: pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, ExecutionStarted: true,
		Cause: &pb.OutcomeCause{Origin: pb.CauseOrigin_CAUSE_ORIGIN_RUNTIME}})
	must(t, err)
	must(t, send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_AttemptOutcome{AttemptOutcome: &pb.AttemptOutcome{
		RecordOwnerEpoch: offer.RecordOwnerEpoch, ControlStreamEpoch: offer.ControlStreamEpoch,
		WorkerBootId: offer.WorkerBootId, RequestId: offer.RequestId, AttemptOrdinal: offer.AttemptOrdinal,
		InvocationSpecDigest: offer.InvocationSpecDigest, OutcomeId: "out-" + requestID,
		OutcomeDigest: digest, OutcomeCanonicalBytes: body,
		PlacementId: offer.PlacementId}}}))
	must(t, r.report(p))
}

// stableRuntimePlacement is the minor-61 Runtime's derivation: placement_id =
// "package-" + sha256(package \0 release)[:24] and model id = "model-" + sha256(slot)[:16],
// so neither moves when the set grows; an entrypoint binds when all its slots are selected.
func stableRuntimePlacement(t *testing.T) func([]byte, string, string) *pb.Placement {
	return func(download []byte, pkg, release string) *pb.Placement {
		placement := podPlacement(download, pkg, release, "")
		seed := sha256.Sum256([]byte(pkg + "\x00" + release))
		placement.PlacementId = "package-" + hex.EncodeToString(seed[:])[:24]
		doc, err := canonical.Read(download, &pb.DownloadDelegation{})
		must(t, err)
		rows := doc.List("models")
		sort.Slice(rows, func(i, j int) bool { return rows[i].Str("slot") < rows[j].Str("slot") })
		placement.Entrypoints = nil
		bound := map[string]*pb.Entrypoint{}
		for _, row := range rows {
			manifest, err := canonical.Raw(row.Str("manifest"))
			must(t, err)
			slotSeed := sha256.Sum256([]byte(row.Str("slot")))
			id := "model-" + hex.EncodeToString(slotSeed[:])[:16]
			placement.Models = append(placement.Models, &pb.Model{Id: id, Repo: row.Str("model"),
				Version: row.Str("release"), Lane: row.Str("lane"),
				Manifest: &pb.Ref{Digest: manifest, Length: 164}})
			name, slot, _ := strings.Cut(row.Str("slot"), ".models.")
			if bound[name] == nil {
				bound[name] = &pb.Entrypoint{Name: name}
				placement.Entrypoints = append(placement.Entrypoints, bound[name])
			}
			bound[name].Slots = append(bound[name].Slots, &pb.Slot{Slot: slot, ReferenceModelId: id,
				Components: []*pb.Component{{Component: "dit", ModelId: id}}})
		}
		return placement
	}
}

// h3Queue is one rental running the H3 entrypoints on the minor-61 Runtime model.
type h3Queue struct {
	t       *testing.T
	pod     *fakePod
	runtime *podRuntime
	o       *owner
	n       int
	gate    chan struct{} // closed: the held preparation may finish
	holdAt  int           // the prepare call (1-based) that waits on gate; 0 holds none
	// named submits as `--rental <name>` does: requested, not pinned, until the fleet's
	// placement decision pins the ladder rung.
	named bool
}

const h3Package = "paul/minimax-h3"

func newH3Queue(t *testing.T, name string, slots uint32, holdAt int, named bool) *h3Queue {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	q := &h3Queue{t: t, runtime: newPodRuntime(), gate: make(chan struct{}), holdAt: holdAt, named: named}
	q.pod = &fakePod{controlKey: public, slots: slots, runtime61: q.runtime,
		preparedPlacement: stableRuntimePlacement(t)}
	q.pod.prepareEvent = func(event *pb.PrepareEvent) {
		if event.Stage != pb.PrepareStage_PREPARE_STAGE_PREPARING {
			return
		}
		q.pod.mu.Lock()
		n := len(q.pod.prepares)
		q.pod.mu.Unlock()
		if n == q.holdAt {
			<-q.gate
		}
	}
	t.Cleanup(q.release)
	connection, _ := startFakePod(t, t.TempDir(), q.pod)
	q.o = hostOwner(t, name, rentalWiring(connection, private), func(opt *orchestrator.Options) {
		opt.RentalFleet = func(records.Request) (string, *exit.Error) { return "one attached rental", nil }
		opt.AcquireManagedRental = func(req records.Request) (orchestrator.PlacementDecision, string, *exit.Error) {
			_, problem := opt.Store.PinRental(req.ID, req.RequestedRental, nil)
			return orchestrator.PlacementDecision{RentalID: req.RequestedRental}, "", problem
		}
	})
	fatal(t, q.o.store.RecordRental(records.Rental{ID: podRental, MachineName: "test-0001", SKU: "h100",
		AcceleratorModel: "H100", AcceleratorCount: 2, HourlyRateUSDMicros: 1, State: "ready", Hub: "fixture"}))
	return q
}

func (q *h3Queue) release() {
	select {
	case <-q.gate:
	default:
		close(q.gate)
	}
}

func h3Model(slot, lane, digit string, shared ...string) orchestrator.ModelRef {
	return orchestrator.ModelRef{Package: h3Package, Slot: slot, SharedSlots: shared, Model: h3Package,
		Release: "1.0.0", Lane: lane, Manifest: "sha256:" + strings.Repeat(digit, 64), ManifestLength: 164,
		Bytes: 20_000_000_000}
}

var (
	fl2vaModels  = []orchestrator.ModelRef{h3Model("fl2va.models.model", "fp8-pruned", "1", "ref2va.models.model")}
	ref2vaModels = []orchestrator.ModelRef{h3Model("ref2va.models.model", "fp8-pruned", "1", "fl2va.models.model")}
)

func turboModels(lora string) []orchestrator.ModelRef {
	base := h3Model("ref2va_turbo.models.base_model", "fp8-adaln-pruned", "2")
	base.Release = "1.0.1"
	adapter := h3Model("ref2va_turbo.models.turbo_lora", "pdd8", lora)
	adapter.Model = h3Package + "-turbo-lora"
	return []orchestrator.ModelRef{base, adapter}
}

func (q *h3Queue) submit(function string, models []orchestrator.ModelRef) string {
	q.t.Helper()
	q.n++
	submission := orchestrator.Submission{
		IdemKey: fmt.Sprintf("h3-%d", q.n), Package: h3Package, Entrypoint: function,
		Release: "1.0.0", Payload: []byte(`{"prompt":"a fox"}`),
		Worker: podRental, Rental: true, RentalRequired: true, Models: models,
	}
	if q.named {
		submission.Worker, submission.RequestedRental = "", podRental
	}
	id, _, problem := q.o.c.Submit(submission)
	fatal(q.t, problem)
	return id
}

func (q *h3Queue) counts() (prepares, desired int, offered []string) {
	q.pod.mu.Lock()
	defer q.pod.mu.Unlock()
	for _, offer := range q.pod.offers {
		offered = append(offered, offer.RequestId)
	}
	return len(q.pod.prepares), len(q.pod.desired), offered
}

func (q *h3Queue) offer(n int) *pb.AttemptOffer {
	q.t.Helper()
	waitUntil(q.t, fmt.Sprintf("offer %d", n), func() bool {
		_, _, offered := q.counts()
		return len(offered) >= n
	})
	q.pod.mu.Lock()
	defer q.pod.mu.Unlock()
	return q.pod.offers[n-1]
}

// quiet asserts nothing more is offered over a few report cadences.
func (q *h3Queue) quiet(offers int, why string) {
	q.t.Helper()
	time.Sleep(1500 * time.Millisecond)
	if _, _, offered := q.counts(); len(offered) != offers {
		q.t.Fatalf("%s: offers %v, want %d", why, offered, offers)
	}
}

// WARM FIRST (run 868, loran). The head needs a selection the rental does not hold; while it
// is prepared, the queued requests the executor already holds run on the warm placement,
// and the grown placement keeps its id. Repeats cost no prepare.
func TestWarmFirstRunsLoadedWorkWhileHeadPrepares(t *testing.T) {
	q := newH3Queue(t, "warm-first", 1, 2, true)
	first := q.submit("fl2va", fl2vaModels)
	warm := q.offer(1)
	turbo := q.submit("ref2va_turbo", turboModels("3"))
	second := q.submit("fl2va", fl2vaModels)
	third := q.submit("fl2va", fl2vaModels)
	waitUntil(t, "the turbo selection is desired while fl2va runs", func() bool {
		prepares, _, _ := q.counts()
		return prepares == 2
	})
	q.runtime.finish(t, q.pod, first)
	if next := q.offer(2); next.RequestId != second || next.PlacementId != warm.PlacementId {
		t.Fatalf("offer 2 went to %s on %s; want the loaded fl2va request %s on %s",
			next.RequestId, next.PlacementId, second, warm.PlacementId)
	}
	q.runtime.finish(t, q.pod, second)
	if next := q.offer(3); next.RequestId != third {
		t.Fatalf("offer 3 went to %s; want the loaded fl2va request %s", next.RequestId, third)
	}
	q.runtime.finish(t, q.pod, third)
	q.quiet(3, "turbo is not ready yet")
	q.release()
	if next := q.offer(4); next.RequestId != turbo || next.PlacementId != warm.PlacementId {
		t.Fatalf("offer 4 went to %s on %s; want turbo %s on the grown placement %s",
			next.RequestId, next.PlacementId, turbo, warm.PlacementId)
	}
	if prepares, _, _ := q.counts(); prepares != 2 {
		t.Fatalf("%d prepares; want one per selection (fl2va, turbo) and none for repeats", prepares)
	}
}

// Five queued calls share one preparation and fill the rental's two-seat lane.
// Each next call is offered while its predecessor still runs. The exact binding
// comes from the pod's prepared and observed set, including when submission held
// an obsolete plan. This is scheduling coverage; real Runtime/GPU qualification
// must additionally prove the cold construction and generated results.
func TestFiveQueuedRequestsReuseObservedRentalBinding(t *testing.T) {
	q := newH3Queue(t, "staged-rental", 2, 0, false)
	q.runtime.mu.Lock()
	q.runtime.lanes = true
	q.runtime.mu.Unlock()
	requests := make([]string, 5)
	for i := range requests {
		id, _, problem := q.o.c.Submit(orchestrator.Submission{
			IdemKey: fmt.Sprintf("queued-%d", i), Package: h3Package, Entrypoint: "fl2va",
			Release: "1.0.0", Payload: []byte(`{"prompt":"a fox"}`),
			PlanID: "sha256:" + strings.Repeat("f", 64),
			Worker: podRental, Rental: true, RentalRequired: true, Models: fl2vaModels,
		})
		fatal(t, problem)
		requests[i] = id
	}
	first, second := q.offer(1), q.offer(2)
	if first.RequestId != requests[0] || second.RequestId != requests[1] {
		t.Fatalf("first two offers %s, %s; queued order %v", first.RequestId, second.RequestId, requests)
	}
	awaitDurable(t, q.o, requests[2], "request.parked")
	if _, _, offered := q.counts(); len(offered) != 2 {
		t.Fatalf("offers %v while lane-0 holds two requests; lane-1 cannot admit this placement", offered)
	}

	for i, id := range requests {
		// The next request is already admitted before the current one finishes.
		if i+1 < len(requests) {
			next := q.offer(i + 2)
			if next.RequestId != requests[i+1] || next.PlacementId != first.PlacementId {
				t.Fatalf("offer %d = %s on %s; want %s on %s", i+2, next.RequestId,
					next.PlacementId, requests[i+1], first.PlacementId)
			}
		}
		q.runtime.mu.Lock()
		held := q.runtime.running[id] != nil
		q.runtime.mu.Unlock()
		if !held {
			t.Fatalf("%s ended before its successor was offered", id)
		}
		q.runtime.finish(t, q.pod, id)
	}
	for _, id := range requests {
		if _, e := q.o.c.AwaitSettled(id, 30*time.Second); e != nil {
			t.Fatalf("%s did not settle: %s", id, briefly(e))
		}
	}
	if prepares, desired, offered := q.counts(); prepares != 1 || desired != 1 || len(offered) != 5 {
		t.Fatalf("five warm calls made %d prepares, %d desired sets and %d offers", prepares, desired, len(offered))
	}
	q.pod.mu.Lock()
	prepared := append([]byte(nil), q.pod.preparedSet...)
	offers := append([]*pb.AttemptOffer(nil), q.pod.offers...)
	q.pod.mu.Unlock()
	set, err := canonical.Read(prepared, &pb.PlacementSet{})
	must(t, err)
	placement := set.List("placements")[0]
	var binding string
	for _, entry := range placement.List("entrypoints") {
		if entry.Str("name") == "fl2va" {
			binding = entry.Str("entrypoint_binding_digest")
		}
	}
	if binding == "" {
		t.Fatal("pod prepared no fl2va binding")
	}
	for _, offer := range offers {
		spec, err := canonical.Read(offer.InvocationSpecCanonicalBytes, &pb.InvocationSpec{})
		must(t, err)
		serving := spec.Sub("serving")
		if serving.Str("entrypoint_binding_digest") != binding || serving.Str("attempt_binding_id") != binding ||
			serving.Str("bindings_digest") != placement.Str("bindings_digest") ||
			offer.PlacementId != placement.Str("placement_id") {
			t.Fatalf("%s offered different identities than the pod prepared: %v", offer.RequestId, serving)
		}
		row, problem := q.o.store.RequestRow(offer.RequestId)
		fatal(t, problem)
		if row == nil || row.PlanID != binding {
			t.Fatalf("%s did not retain the observed binding: %+v", offer.RequestId, row)
		}
	}
}

// THE STARVATION BOUND. A request passed over twice claims the rental: loaded work behind
// it waits until it has run.
func TestStarvationBoundTwo(t *testing.T) {
	q := newH3Queue(t, "starvation-bound", 1, 2, false)
	first := q.submit("fl2va", fl2vaModels)
	q.offer(1)
	turbo := q.submit("ref2va_turbo", turboModels("3"))
	a := q.submit("fl2va", fl2vaModels)
	b := q.submit("fl2va", fl2vaModels)
	c := q.submit("fl2va", fl2vaModels)
	waitUntil(t, "the turbo selection is desired", func() bool {
		prepares, _, _ := q.counts()
		return prepares == 2
	})
	q.runtime.finish(t, q.pod, first)
	q.offer(2)
	q.runtime.finish(t, q.pod, a)
	q.offer(3)
	q.runtime.finish(t, q.pod, b)
	q.quiet(3, "turbo was passed over twice; nothing younger may run ahead of it")
	if _, ok := waitEvent(q.o, turbo+" was passed over 2 time(s)", time.Second); !ok {
		t.Fatalf("the bound was not recorded: %v", q.o.c.Events())
	}
	q.release()
	if next := q.offer(4); next.RequestId != turbo {
		t.Fatalf("offer 4 went to %s; want the starved turbo request %s", next.RequestId, turbo)
	}
	q.runtime.finish(t, q.pod, turbo)
	if next := q.offer(5); next.RequestId != c {
		t.Fatalf("offer 5 went to %s; want %s", next.RequestId, c)
	}
	_, _, offered := q.counts()
	if strings.Join(offered, ",") != strings.Join([]string{first, a, b, turbo, c}, ",") {
		t.Fatalf("dispatch order %v", offered)
	}
}

// HOLDS = OBSERVED. A newer desired set the worker has not accepted holds nothing: another
// entrypoint the set the worker DOES hold binds runs there, and the unaccepted selection
// is not treated as held.
func TestHoldsReadsObservedSet(t *testing.T) {
	q := newH3Queue(t, "holds-observed", 2, 0, false)
	q.runtime.hold = func(d *pb.DesiredWorkerState) bool {
		q.pod.mu.Lock()
		defer q.pod.mu.Unlock()
		return len(q.pod.desired) > 1
	}
	q.submit("fl2va", fl2vaModels)
	warm := q.offer(1)
	turbo := q.submit("ref2va_turbo", turboModels("3"))
	waitUntil(t, "the grown set is sent and left unaccepted", func() bool {
		_, desired, _ := q.counts()
		return desired == 2
	})
	ref2va := q.submit("ref2va", ref2vaModels)
	if next := q.offer(2); next.RequestId != ref2va || next.PlacementId != warm.PlacementId {
		t.Fatalf("offer 2 went to %s on %s; want ref2va %s on the observed placement %s",
			next.RequestId, next.PlacementId, ref2va, warm.PlacementId)
	}
	again := q.submit("ref2va_turbo", turboModels("3"))
	q.quiet(2, "the turbo selection is only desired, never observed")
	if prepares, desired, _ := q.counts(); prepares != 2 || desired != 2 {
		t.Fatalf("%d prepares and %d desired states; a repeat of the pending selection prepares nothing",
			prepares, desired)
	}
	for _, id := range []string{turbo, again} {
		if o := q.o.c.QueuePosition(id); o == 0 {
			t.Fatalf("%s left the queue without an observed placement", id)
		}
	}
}

// NO EARLY PREPARE ON SUBMIT. A submission prepares nothing itself: the rental takes one
// desire at a time, the oldest selection it does not hold, and the next only when that
// one has ended; a held selection never prepares.
func TestNoEarlyPrepareOnSubmit(t *testing.T) {
	q := newH3Queue(t, "no-early-prepare", 1, 2, true)
	q.submit("fl2va", fl2vaModels)
	q.offer(1)
	q.submit("ref2va_turbo", turboModels("3"))
	waitUntil(t, "the first unheld selection is desired", func() bool {
		prepares, _, _ := q.counts()
		return prepares == 2
	})
	q.submit("ref2va_turbo", turboModels("4"))
	q.submit("fl2va", fl2vaModels)
	time.Sleep(1500 * time.Millisecond)
	if prepares, desired, _ := q.counts(); prepares != 2 || desired != 1 {
		t.Fatalf("submissions caused %d prepares and %d desired states; want the one in flight", prepares, desired)
	}
	q.release()
	waitUntil(t, "the next unheld selection is desired after the first ends", func() bool {
		prepares, _, _ := q.counts()
		return prepares == 3
	})
}
