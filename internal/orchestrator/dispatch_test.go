package orchestrator

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy-creator-v2/internal/canonical"
	"github.com/cozy-creator/cozy-creator-v2/internal/home"
	"github.com/cozy-creator/cozy-creator-v2/internal/media"
	"github.com/cozy-creator/cozy-creator-v2/internal/records"
	"github.com/cozy-creator/cozy-creator-v2/internal/secret"
	pb "github.com/cozy-creator/cozy-creator-v2/protocol/cozy/worker/v1"
)

// This is the executable regression for the pre-offer boundary: the first media upload
// fails after the worker seat and durable ordinal are reserved; the same worker is then
// used immediately, without a new Report, and the next ordinal is offered successfully.
func TestDispatchPreparationFailureAbortsAndReturnsSeat(t *testing.T) {
	var fail atomic.Bool
	var failDrop atomic.Bool
	var drops atomic.Int64
	fail.Store(true)
	failDrop.Store(true)
	peer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodPost && strings.HasPrefix(r.URL.Path, "/v1/outputs/"):
			_ = json.NewEncoder(w).Encode(map[string]any{"dir": "/pod/out"})
		case r.Method == http.MethodPut && strings.HasPrefix(r.URL.Path, "/v1/inputs/"):
			data, _ := io.ReadAll(r.Body)
			if fail.Load() {
				w.WriteHeader(http.StatusInsufficientStorage)
				_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]string{
					"code": "media.quota_exhausted", "message": "full", "remedy": "drop old attempts",
				}})
				return
			}
			sum := sha256.Sum256(data)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"path": "/pod/input", "digest": "sha256:" + hex.EncodeToString(sum[:]),
			})
		case r.Method == http.MethodDelete && strings.HasPrefix(r.URL.Path, "/v1/attempts/"):
			drops.Add(1)
			if failDrop.Load() {
				w.WriteHeader(http.StatusServiceUnavailable)
				_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]string{
					"code": "media.drop_failed", "message": "busy", "remedy": "retry",
				}})
				return
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"slot": strings.TrimPrefix(r.URL.Path, "/v1/attempts/")})
		default:
			http.NotFound(w, r)
		}
	}))
	defer peer.Close()

	root := t.TempDir()
	layout, e := home.Open(root)
	if e != nil {
		t.Fatal(e)
	}
	store, e := records.Open(layout.DB)
	if e != nil {
		t.Fatal(e)
	}
	defer store.Close()
	client, e := media.Dial(media.Spec{
		Addr: strings.TrimPrefix(peer.URL, "http://"), Token: secret.New("test-token"),
	}, time.Second, 1<<20)
	if e != nil {
		t.Fatal(e)
	}
	const instance, boot, endpoint, planID = "ins-test", "boot-test", "org/model", "sha256:plan"
	if e := store.AttachWorker(records.WorkerProcess{
		InstanceID: instance, Endpoint: endpoint, ReleaseID: "release", WorkerID: instance,
	}); e != nil {
		t.Fatal(e)
	}
	c, e := Open(Options{Layout: layout, Store: store,
		EnvironmentSpecDigest: "sha256:environment", ConfigDigest: "sha256:config", MaxOutputMiB: 1})
	if e != nil {
		t.Fatal(e)
	}
	w := &worker{
		instanceID: instance, bootID: boot, placementID: "plc-test", media: client,
		spec:         WorkerLaunchSpec{Placement: DesiredPlacement{Endpoint: endpoint, ReleaseID: "release"}},
		serving:      pb.ServingState_SERVING_STATE_DISPATCHABLE,
		admission:    pb.AdmissionState_ADMISSION_STATE_OPEN,
		dispatchable: map[string]bool{planID: true}, reportedSlots: 1, slots: 1,
	}
	sess := &session{bootID: boot, instanceID: instance, generation: 1, out: make(chan *pb.RecordOwnerFrame, 1)}
	c.workers[instance], c.sessions[boot] = w, sess
	req, _, e := store.Submit(records.Request{
		ID: "req-test", IdemKey: "idem-test", BodyDigest: "sha256:body",
		Endpoint: endpoint, Entrypoint: "generate", PlanID: planID,
		Payload: []byte(`{"prompt":"test"}`), Outputs: "image",
	})
	if e != nil {
		t.Fatal(e)
	}

	if _, e := c.dispatch(req); e == nil || e.ErrName() != "media.quota_exhausted" {
		t.Fatalf("first dispatch error = %v, want media.quota_exhausted", e)
	}
	attempts, e := store.Attempts(req.ID)
	if e != nil || len(attempts) != 1 || attempts[0].State != "dispatch_aborted" {
		t.Fatalf("attempts after failed preparation = %#v, %v", attempts, e)
	}
	row, e := store.RequestRow(req.ID)
	if e != nil || row == nil || row.State != "submitted" {
		t.Fatalf("request after failed preparation = %#v, %v", row, e)
	}
	if w.slots != 1 || w.reservedSlots != 0 {
		t.Fatalf("seat after failed preparation = slots %d, reserved %d", w.slots, w.reservedSlots)
	}
	if drops.Load() != 1 {
		t.Fatalf("partial remote grant cleanup attempts = %d, want 1", drops.Load())
	}
	select {
	case <-sess.out:
		t.Fatal("worker received an offer for the aborted dispatch")
	default:
	}

	failDrop.Store(false)
	c.retryMediaCleanup(w)
	if drops.Load() != 2 {
		t.Fatalf("durable cleanup retry count = %d, want 2", drops.Load())
	}
	fail.Store(false)
	attempt, e2 := c.dispatch(req)
	if e2 != nil || attempt != 2 {
		t.Fatalf("retry dispatch = attempt %d, %v; want ordinal 2 without a new Report", attempt, e2)
	}
	select {
	case frame := <-sess.out:
		if frame.GetAttemptOffer() == nil || frame.GetAttemptOffer().AttemptOrdinal != 2 {
			t.Fatalf("retry frame = %#v", frame)
		}
	default:
		t.Fatal("retry emitted no AttemptOffer")
	}
	w.observeSlots(1) // a report that raced the outbound offer is not a causal answer
	if w.slots != 0 || w.reservedSlots != 1 {
		t.Fatalf("racing Report reopened an offered seat: slots %d, reserved %d", w.slots, w.reservedSlots)
	}
	c.settleDispatch(req.ID, attempt, true)
	if w.slots != 0 || w.reservedSlots != 0 {
		t.Fatalf("accepted offer did not consume its reported seat: slots %d, reserved %d", w.slots, w.reservedSlots)
	}
	c.cleanupAttempt(req, attempt, w, true) // production calls this only after mirror + OutcomeAck
	if drops.Load() != 3 {
		t.Fatalf("terminal remote cleanup count = %d, want rollback, retry, and post-ack cleanup", drops.Load())
	}
}

func TestLocalWindowsGrantRefusesBeforeWritingPaths(t *testing.T) {
	e := localGrantSupport("windows")
	if e == nil || e.ErrName() != "local_file_grant_unsupported" {
		t.Fatalf("Windows local-grant verdict = %v", e)
	}
	if e := localGrantSupport("linux"); e != nil {
		t.Fatalf("Linux local-grant verdict = %v", e)
	}
}

func TestWorkerRefusalOriginHardcut(t *testing.T) {
	if !requeueable("REFUSED", "NO_CAPACITY", "WORKER", false) {
		t.Fatal("a pre-execution WORKER refusal did not earn a new ordinal")
	}
	if requeueable("REFUSED", "NO_CAPACITY", "SUPERVISOR", false) {
		t.Fatal("the retired SUPERVISOR origin remained an alias")
	}
	if requeueable("REFUSED", "NO_CAPACITY", "WORKER", true) {
		t.Fatal("a worker refusal that claims execution started was requeued")
	}
}

func TestUnauthorizedRefusalCannotReturnAnotherAssignmentsSeat(t *testing.T) {
	tests := []struct {
		name        string
		sessionBoot string
		outcomeSpec []byte
	}{
		{name: "wrong session", sessionBoot: "boot-intruder"},
		{name: "wrong invocation digest", sessionBoot: "boot-owner", outcomeSpec: []byte("wrong-spec")},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			layout, e := home.Open(t.TempDir())
			if e != nil {
				t.Fatal(e)
			}
			store, e := records.Open(layout.DB)
			if e != nil {
				t.Fatal(e)
			}
			defer store.Close()

			const instance, ownerBoot = "ins-owner", "boot-owner"
			if e := store.AttachWorker(records.WorkerProcess{
				InstanceID: instance, Endpoint: "org/model", ReleaseID: "release", WorkerID: instance,
			}); e != nil {
				t.Fatal(e)
			}
			req, _, e := store.Submit(records.Request{
				ID: "req-authority", IdemKey: "idem-authority", BodyDigest: "sha256:body",
				Endpoint: "org/model", Entrypoint: "generate", PlanID: "plan", Payload: []byte(`{}`),
			})
			if e != nil {
				t.Fatal(e)
			}
			assigned := sha256.Sum256([]byte("assigned-spec"))
			spelled, _ := canonical.Spell(assigned[:])
			ordinal, e := store.Dispatch(records.Attempt{
				RequestID: req.ID, InstanceID: instance, SessionID: ownerBoot,
				InvocationDigest: spelled, InvocationCanonical: []byte("assigned-spec"),
			})
			if e != nil {
				t.Fatal(e)
			}
			if e := store.OfferDispatch(req.ID, ordinal, ownerBoot); e != nil {
				t.Fatal(e)
			}

			c, e := Open(Options{Layout: layout, Store: store})
			if e != nil {
				t.Fatal(e)
			}
			w := &worker{instanceID: instance, bootID: ownerBoot, reportedSlots: 1, reservedSlots: 1}
			w.observeSlots(1)
			c.workers[instance] = w
			c.offers[key(req.ID, uint64(ordinal))] = &dispatchReservation{worker: w}

			spec := assigned[:]
			if tt.outcomeSpec != nil {
				wrong := sha256.Sum256(tt.outcomeSpec)
				spec = wrong[:]
			}
			outcome := refusedOutcome(t, req.ID, uint64(ordinal), spec)
			c.onOutcome(&session{
				bootID: tt.sessionBoot, instanceID: instance, generation: 1,
				out: make(chan *pb.RecordOwnerFrame, 1),
			}, outcome)

			if _, ok := c.offers[key(req.ID, uint64(ordinal))]; !ok {
				t.Fatal("unauthorized outcome removed the live dispatch reservation")
			}
			if w.reservedSlots != 1 || w.slots != 0 {
				t.Fatalf("unauthorized outcome reopened the seat: slots=%d reserved=%d", w.slots, w.reservedSlots)
			}
			attempt, e := store.AttemptRow(req.ID, ordinal)
			if e != nil || attempt == nil || attempt.State != "offered" {
				t.Fatalf("unauthorized outcome changed durable attempt: %#v, %v", attempt, e)
			}
		})
	}
}

func refusedOutcome(t *testing.T, requestID string, ordinal uint64, spec []byte) *pb.AttemptOutcome {
	t.Helper()
	spelled, e := canonical.Spell(spec)
	if e != nil {
		t.Fatal(e)
	}
	body := &pb.AttemptOutcomeBody{
		RequestId: requestID, AttemptOrdinal: ordinal, InvocationSpecDigest: spelled,
		Status: pb.OutcomeStatus_OUTCOME_STATUS_REFUSED, SafeMessage: "no capacity",
		ExecutionStarted: false,
		Cause: &pb.OutcomeCause{
			Code: pb.CauseCode_CAUSE_CODE_NO_CAPACITY, Origin: pb.CauseOrigin_CAUSE_ORIGIN_WORKER,
			Detail: "refused before execution",
		},
	}
	data, digest, e := canonical.Identity(body)
	if e != nil {
		t.Fatal(e)
	}
	return &pb.AttemptOutcome{
		RequestId: requestID, AttemptOrdinal: ordinal, InvocationSpecDigest: spec,
		OutcomeId: "out-refused", OutcomeDigest: digest, OutcomeCanonicalBytes: data,
	}
}

func TestTerminalCleanupRetainsSharedAssetUntilLastLiveRequest(t *testing.T) {
	layout, e := home.Open(t.TempDir())
	if e != nil {
		t.Fatal(e)
	}
	store, e := records.Open(layout.DB)
	if e != nil {
		t.Fatal(e)
	}
	defer store.Close()
	c, e := Open(Options{Layout: layout, Store: store})
	if e != nil {
		t.Fatal(e)
	}
	digest := "sha256:" + strings.Repeat("0", 64)
	assetPath := layout.InputAsset(digest)
	if err := os.WriteFile(assetPath, []byte("shared"), 0o600); err != nil {
		t.Fatal(err)
	}
	asset := records.AssetBinding{FieldPath: "image", LocalPath: assetPath, Digest: digest, Length: 6}
	submit := func(id string) records.Request {
		r, _, e := store.Submit(records.Request{
			ID: id, IdemKey: "idem-" + id, BodyDigest: "sha256:" + id,
			Endpoint: "org/model", Entrypoint: "generate", PlanID: "plan", Payload: []byte(`{}`),
			Assets: []records.AssetBinding{asset},
		})
		if e != nil {
			t.Fatal(e)
		}
		return r
	}
	first, second := submit("first"), submit("second")
	firstInputs := filepath.Join(layout.AttemptDir(first.ID, 1), "in")
	if err := os.MkdirAll(firstInputs, 0o755); err != nil {
		t.Fatal(err)
	}
	if e := store.SettleRequest(first.ID, "succeeded"); e != nil {
		t.Fatal(e)
	}
	c.cleanupAttempt(first, 1, nil, true)
	if _, err := os.Stat(firstInputs); !os.IsNotExist(err) {
		t.Fatalf("terminal local attempt input directory remains: %v", err)
	}
	if _, err := os.Stat(assetPath); err != nil {
		t.Fatalf("asset shared by a live request was removed: %v", err)
	}
	if e := store.SettleRequest(second.ID, "succeeded"); e != nil {
		t.Fatal(e)
	}
	c.cleanupAttempt(second, 1, nil, true)
	if _, err := os.Stat(assetPath); !os.IsNotExist(err) {
		t.Fatalf("asset remains after its last live request settled: %v", err)
	}
}

func TestRestartAbortsPreparedAttemptAbsentFromSnapshot(t *testing.T) {
	peer, drops := cleanupPeer(t)
	root := t.TempDir()
	layout, e := home.Open(root)
	if e != nil {
		t.Fatal(e)
	}
	store, e := records.Open(layout.DB)
	if e != nil {
		t.Fatal(e)
	}
	client, e := media.Dial(media.Spec{
		Addr: strings.TrimPrefix(peer.URL, "http://"), Token: secret.New("test-token"),
	}, time.Second, 1<<20)
	if e != nil {
		t.Fatal(e)
	}
	const instance = "ins-restart-preparing"
	if e := store.AttachWorker(records.WorkerProcess{InstanceID: instance, Endpoint: "org/model",
		ReleaseID: "release", WorkerID: instance}); e != nil {
		t.Fatal(e)
	}
	req, _, e := store.Submit(records.Request{ID: "req-preparing", IdemKey: "idem-preparing",
		BodyDigest: "sha256:body", Endpoint: "org/model", Entrypoint: "generate",
		PlanID: "plan", Payload: []byte(`{"prompt":"test"}`), Outputs: "image"})
	if e != nil {
		t.Fatal(e)
	}
	ordinal, e := store.Dispatch(records.Attempt{RequestID: req.ID, InstanceID: instance,
		SessionID: "boot-before-crash", InvocationDigest: "sha256:spec", InvocationCanonical: []byte("spec")})
	if e != nil {
		t.Fatal(e)
	}
	before, e := Open(Options{Layout: layout, Store: store})
	if e != nil {
		t.Fatal(e)
	}
	w := &worker{instanceID: instance, media: client}
	if _, e := before.remoteGrant(req, uint64(ordinal), w); e != nil {
		t.Fatal(e)
	}
	store.Close()

	store, e = records.Open(layout.DB)
	if e != nil {
		t.Fatal(e)
	}
	defer store.Close()
	after, e := Open(Options{Layout: layout, Store: store})
	if e != nil {
		t.Fatal(e)
	}
	after.workers[instance] = w
	after.reconcileSnapshotAbsence(w, map[string]bool{})
	after.retryMediaCleanup(w)
	attempt, e := store.AttemptRow(req.ID, ordinal)
	if e != nil || attempt == nil || attempt.State != "dispatch_aborted" {
		t.Fatalf("reconciled attempt = %#v, %v", attempt, e)
	}
	row, e := store.RequestRow(req.ID)
	if e != nil || row == nil || row.State != "submitted" || after.QueuePosition(req.ID) != 1 {
		t.Fatalf("reconciled request = %#v, queue position %d, %v", row, after.QueuePosition(req.ID), e)
	}
	if drops.Load() != 1 {
		t.Fatalf("remote rollback deletes = %d, want 1", drops.Load())
	}
}

func TestRestartClosesCompactedTerminalBeforeCleanup(t *testing.T) {
	peer, drops := cleanupPeer(t)
	root := t.TempDir()
	layout, e := home.Open(root)
	if e != nil {
		t.Fatal(e)
	}
	store, e := records.Open(layout.DB)
	if e != nil {
		t.Fatal(e)
	}
	client, e := media.Dial(media.Spec{
		Addr: strings.TrimPrefix(peer.URL, "http://"), Token: secret.New("test-token"),
	}, time.Second, 1<<20)
	if e != nil {
		t.Fatal(e)
	}
	const instance = "ins-restart-terminal"
	if e := store.AttachWorker(records.WorkerProcess{InstanceID: instance, Endpoint: "org/model",
		ReleaseID: "release", WorkerID: instance}); e != nil {
		t.Fatal(e)
	}
	assetDigest := "sha256:" + strings.Repeat("1", 64)
	assetPath := layout.InputAsset(assetDigest)
	if err := os.WriteFile(assetPath, []byte("asset"), 0o600); err != nil {
		t.Fatal(err)
	}
	req, _, e := store.Submit(records.Request{ID: "req-terminal", IdemKey: "idem-terminal",
		BodyDigest: "sha256:body", Endpoint: "org/model", Entrypoint: "generate",
		PlanID: "plan", Payload: []byte(`{}`), Assets: []records.AssetBinding{{
			FieldPath: "first_frame", LocalPath: assetPath, Digest: assetDigest, Length: 5,
		}}})
	if e != nil {
		t.Fatal(e)
	}
	ordinal, e := store.Dispatch(records.Attempt{RequestID: req.ID, InstanceID: instance,
		SessionID: "boot-before-crash", InvocationDigest: "sha256:spec", InvocationCanonical: []byte("spec")})
	if e != nil {
		t.Fatal(e)
	}
	if e := store.OfferDispatch(req.ID, ordinal, "boot-before-crash"); e != nil {
		t.Fatal(e)
	}
	if _, e := store.AcceptTerminal(records.Terminal{RequestID: req.ID, Attempt: ordinal,
		SessionID: "boot-before-crash", InvocationDigest: "sha256:spec",
		TerminalID: "outcome", TerminalDigest: "sha256:outcome", Status: "SUCCEEDED",
		RequestState: "succeeded"}); e != nil {
		t.Fatal(e)
	}
	store.Close()

	store, e = records.Open(layout.DB)
	if e != nil {
		t.Fatal(e)
	}
	defer store.Close()
	after, e := Open(Options{Layout: layout, Store: store})
	if e != nil {
		t.Fatal(e)
	}
	w := &worker{instanceID: instance, media: client}
	continuations := after.reconcileSnapshotAbsence(w, map[string]bool{})
	if len(continuations) != 1 {
		t.Fatalf("post-ack continuations = %d, want 1", len(continuations))
	}
	for _, continuation := range continuations {
		after.afterAck(continuation.request, continuation.attempt, w)
	}
	attempt, e := store.AttemptRow(req.ID, ordinal)
	if e != nil || attempt == nil || attempt.State != "closed" {
		t.Fatalf("reconciled terminal = %#v, %v", attempt, e)
	}
	if drops.Load() != 1 {
		t.Fatalf("terminal cleanup deletes = %d, want 1", drops.Load())
	}
	if _, err := os.Stat(assetPath); !os.IsNotExist(err) {
		t.Fatalf("settled request asset remains: %v", err)
	}
}

func cleanupPeer(t *testing.T) (*httptest.Server, *atomic.Int64) {
	t.Helper()
	var drops atomic.Int64
	peer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodPost && strings.HasPrefix(r.URL.Path, "/v1/outputs/"):
			_ = json.NewEncoder(w).Encode(map[string]any{"dir": "/pod/out"})
		case r.Method == http.MethodPut && strings.HasPrefix(r.URL.Path, "/v1/inputs/"):
			data, _ := io.ReadAll(r.Body)
			sum := sha256.Sum256(data)
			_ = json.NewEncoder(w).Encode(map[string]any{"path": "/pod/input",
				"digest": "sha256:" + hex.EncodeToString(sum[:]), "length": len(data)})
		case r.Method == http.MethodDelete && strings.HasPrefix(r.URL.Path, "/v1/attempts/"):
			drops.Add(1)
			_ = json.NewEncoder(w).Encode(map[string]any{"deleted": true})
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(peer.Close)
	return peer, &drops
}

func TestPickHonorsExactInstallWithEqualPlanID(t *testing.T) {
	const planID = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	c := &Orchestrator{workers: map[string]*worker{}, sessions: map[string]*session{}}
	makeWorker := func(id, install, boot string) *worker {
		w := newWorker(id, WorkerLaunchSpec{Placement: DesiredPlacement{
			Endpoint: "org/ep", InstallID: install,
		}})
		w.bootID = boot
		w.serving = pb.ServingState_SERVING_STATE_DISPATCHABLE
		w.dispatchable[planID] = true
		w.admission = pb.AdmissionState_ADMISSION_STATE_OPEN
		w.observeSlots(1)
		c.workers[id] = w
		c.sessions[boot] = &session{bootID: boot, instanceID: id,
			out: make(chan *pb.RecordOwnerFrame, 1)}
		return w
	}
	old := makeWorker("old", "install-old", "boot-old")
	want := makeWorker("wanted", "install-wanted", "boot-wanted")
	got, _, _, reservation, problem := c.pick(records.Request{
		Endpoint: "org/ep", PlanID: planID, InstallID: "install-wanted",
	})
	if problem != nil || got != want || got == old {
		t.Fatalf("picked=%v want=%v problem=%v", got, want, problem)
	}
	c.releaseDispatch(reservation)
}

func TestLocalSnapshotHasNoRemoteMediaCleanup(t *testing.T) {
	// A local worker deliberately has no media client. Snapshot recovery shares the
	// cleanup path with remote workers, so this call is the nil-client regression arm.
	(&Orchestrator{}).retryMediaCleanup(&worker{instanceID: "local"})
}
