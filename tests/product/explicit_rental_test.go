package producttest

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func TestNamedRentalCLIUsesActualInventoryAndNeverAcquires(t *testing.T) {
	const current = "pr-11111111111111111111"
	const old = "pr-22222222222222222222"
	var inventoryReads atomic.Int32
	root, mu, posts, _, _ := runModelCatalog(t, func(mux *http.ServeMux, detail *hub.PackageReleaseDetail) {
		detail.ExecutionRequirements = []string{"cozy-runtime[media]>=0.15.0,<1"}
		mux.HandleFunc("GET /v1/rentals/{id}", func(w http.ResponseWriter, r *http.Request) {
			id := r.PathValue("id")
			name, ok := map[string]string{current: "isao", old: "giriko"}[id]
			if !ok {
				http.NotFound(w, r)
				return
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"rental_id": id, "name": name, "state": "ready", "development": false,
				"requested_accelerator_model": "CPU", "accelerator_count": 1,
				"hourly_rate_usd_micros": 100000,
			})
		})
	})
	st, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	// The rentals were bought from the catalog's hub; a named rental's hub serves its run.
	hubURL := configuredHub(t, root)
	for id, name := range map[string]string{current: "isao", old: "giriko"} {
		fatal(t, st.RecordRental(records.Rental{ID: id, MachineName: name, AcceleratorModel: "CPU", AcceleratorCount: 1, State: "ready", HourlyRateUSDMicros: 100000, Address: "127.0.0.1:1", CertPath: "unused", Hub: hubURL}))
	}
	st.Close()
	installedHere(t, root, hubURL, "proof/quantize", "1.0.0")
	args := []string{"run", "proof/quantize/quantize", "steps=7", "model.dits=proof/source@1.0.0/bf16", "model.shared=proof/source@1.0.0/bf16", "--json", "--full"}
	for _, name := range []string{"isao", current} {
		request, _, out := submitRun(t, root, "named-"+name, append(append([]string{}, args...), "--rental="+name)...)
		if request == nil || request.RequestedRental != current {
			t.Fatalf("named rental %s: %+v %s", name, request, out)
		}
	}
	request, _, out := submitRun(t, root, "named-giriko", append(append([]string{}, args...), "--rental=giriko")...)
	if request == nil || request.RequestedRental != old {
		t.Fatalf("package-private Runtime was constrained by image inventory: %+v %s", request, out)
	}
	// Replaying an accepted request preserves its selected identity after the rental
	// ends; it does not need a fresh inventory read merely to retrieve history.
	st, problem = records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	_, _, problem = st.Submit(records.Request{ID: "prior", IdemKey: "prior", BodyDigest: childDigest("f"), Package: "proof/quantize", Entrypoint: "quantize", Payload: []byte("{}"), Rental: true, RequestedRental: current})
	fatal(t, problem)
	_, problem = st.FailQueuedRequest("prior", map[string]any{"error_type": "proof", "error": "stopped"})
	fatal(t, problem)
	_, problem = st.ForgetRental(current)
	fatal(t, problem)
	st.Close()
	replayReads := inventoryReads.Load()
	code, out := runCozy(t, root, append(append([]string{}, args...), "--rental=isao", "--idempotency-key=prior")...)
	if strings.Contains(out, "rental.selection_unavailable") || inventoryReads.Load() != replayReads {
		t.Fatalf("replay required a released rental: %d %s", code, out)
	}
	before := inventoryReads.Load()
	code, out = runCozy(t, root, append(append([]string{}, args...), "--rental=unknown")...)
	if code == 0 || !strings.Contains(out, "rental.selection_unavailable") || inventoryReads.Load() != before {
		t.Fatalf("unknown rental consulted remote: %d %s", code, out)
	}
	code, out = runCozy(t, root, append(append([]string{}, args...), "--rental=isao", "--rental-only")...)
	if code == 0 {
		t.Fatalf("conflicting rental selection accepted: %s", out)
	}
	code, out = runCozy(t, root, "run", "proof/quantize/quantize", "--rental=")
	if code == 0 || !strings.Contains(out, "--rental-only") {
		t.Fatalf("empty rental silently became local: %d %s", code, out)
	}
	code, out = runCozy(t, root, "run", "proof/quantize/quantize", "--rental")
	if code == 0 || !strings.Contains(out, "--rental-only") {
		t.Fatalf("bare old flag lacks guidance: %d %s", code, out)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(*posts) != 0 {
		t.Fatal("existing-rental selection authorized a purchase")
	}
}

func TestRequestedRentalSurvivesRecordsAndRejectsReassignment(t *testing.T) {
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	st, problem := records.Open(path)
	fatal(t, problem)
	_, _, problem = st.Submit(records.Request{ID: "fixed", IdemKey: "fixed", BodyDigest: "sha256:" + strings.Repeat("1", 64), Package: "proof/model", Entrypoint: "run", Payload: []byte("{}"), Rental: true, RentalRequired: true, RequestedRental: "wanted"})
	fatal(t, problem)
	pinned, problem := st.PinRental("fixed", "other", nil)
	fatal(t, problem)
	if pinned {
		t.Fatal("affinity assigned to another rental")
	}
	pinned, problem = st.PinRental("fixed", "wanted", nil)
	fatal(t, problem)
	if !pinned {
		t.Fatal("wanted rental was not assigned")
	}
	unpinned, problem := st.UnpinRentalWork("fixed", "wanted")
	fatal(t, problem)
	if unpinned {
		t.Fatal("fixed rental silently became automatic")
	}
	st.Close()
	st, problem = records.Open(path)
	fatal(t, problem)
	defer st.Close()
	row, problem := st.RequestRow("fixed")
	fatal(t, problem)
	if row.RequestedRental != "wanted" || row.Worker != "wanted" {
		t.Fatalf("affinity lost after reopen: %+v", row)
	}
}

func TestRentalNewIsTheOnlyCreationCommand(t *testing.T) {
	root, _, _, _, _ := runModelCatalog(t)
	code, out := runCozy(t, root, "rental", "new", "--json")
	if code != 0 || !strings.Contains(out, "cpu") {
		t.Fatalf("rental new catalog: %d %s", code, out)
	}
	for _, args := range [][]string{{"--help"}, {"rental", "--help"}} {
		code, out = runCozy(t, root, args...)
		if code != 0 || !strings.Contains(out, "rental new") || strings.Contains(out, "\n  rent ") {
			t.Fatalf("canonical creation command is not advertised by %v: %d %s", args, code, out)
		}
	}
	code, out = runCozy(t, root, "rental", "new", "--help")
	if code != 0 || !strings.Contains(out, "<machine-slug>") || !strings.Contains(out, "h100-sxm5-80gb") {
		t.Fatalf("rental new does not describe its machine slug: %d %s", code, out)
	}
	// `cozy rent` is the rental group under another name (Paul, 2026-09-24), never a
	// second creation verb: `rent <sku>` is refused, `rent new` is `rental new`.
	code, out = runCozy(t, root, "rent", "cpu", "--json")
	if code == 0 {
		t.Fatalf("`cozy rent <sku>` became a creation shorthand: %s", out)
	}
	code, out = runCozy(t, root, "rent", "new", "--json")
	if code != 0 || !strings.Contains(out, "cpu") {
		t.Fatalf("`cozy rent new` is not `cozy rental new`: %d %s", code, out)
	}
}

func TestAutomaticAndNamedRentalChoicesAllowPrivateSDKVersions(t *testing.T) {
	h := newLadderHub(t)
	h.bind(goodLadder())
	h.runtimeVersions = map[string]string{"pr-oldruntime": "0.1.0"}
	root, _ := placementRoot(t, h, "balanced")
	cert := filepath.Join(root, "fixture.pem")
	must(t, os.WriteFile(cert, []byte("fixture"), 0600))
	st, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	for id, name := range map[string]string{"pr-oldruntime": "giriko", "pr-newruntime": "isao"} {
		row := records.Rental{ID: id, MachineName: name, SKU: "h100-80", AcceleratorModel: h100SXM, AcceleratorCount: 1, HourlyRateUSDMicros: 2490000, State: "ready", Address: "127.0.0.1:1", CertPath: cert, Hub: h.server.URL}
		fatal(t, st.RecordRental(row))
		h.addReady(id, name, h100SXM, row.HourlyRateUSDMicros)
	}
	st.Close()
	installedHere(t, root, h.server.URL, ladderPackage, "1.0.0")
	startDaemonProcess(t, root)
	for _, arm := range []struct{ key, flag string }{{"automatic-runtime", "--rental-only"}, {"named-runtime", "--rental=giriko"}} {
		_, out := runCozy(t, root, "run", ladderPackage+"/generate", "steps=1", arm.flag, "--json", "--idempotency-key", arm.key)
		st, problem = records.Open(filepath.Join(root, "creator.sqlite"))
		fatal(t, problem)
		row, problem := st.RequestByIdempotencyKey(arm.key)
		fatal(t, problem)
		if row == nil || row.Worker != "pr-oldruntime" || strings.Contains(out, "rental.runtime_incompatible") {
			t.Fatalf("%s constrained a package SDK to the worker image: %+v %s", arm.key, row, out)
		}
		if arm.key == "named-runtime" && row.RequestedRental != "pr-oldruntime" {
			t.Fatal("explicit request lost durable affinity")
		}
		st.Close()
	}
	if asks := h.postedSKUs(); len(asks) != 0 {
		t.Fatalf("compatible attached rental lost to purchase: %v", asks)
	}
}

func TestRequestedRentalFlowsToChildAndRetainedRetry(t *testing.T) {
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	st, problem := records.Open(path)
	fatal(t, problem)
	defer st.Close()
	fatal(t, st.RecordRental(records.Rental{ID: "wanted", MachineName: "isao", AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 100000, State: "ready"}))
	parent, _, problem := st.Submit(records.Request{ID: "parent", IdemKey: "parent", BodyDigest: childDigest("a"), Package: "local/script", Entrypoint: "main", Kind: "job", Payload: []byte("{}"), RetainWork: true, Rental: true, RentalRequired: true, Worker: "wanted", RequestedRental: "wanted"})
	fatal(t, problem)
	fatal(t, st.SpawnWorker(records.WorkerProcess{InstanceID: "private-worker", Package: "local/script", WorkerID: "worker", Devices: []string{"cpu"}}))
	parent = offerChildParent(t, st, parent)
	call := records.Request{ID: "child", IdemKey: "child", BodyDigest: childDigest("b"), Package: "local/op", Entrypoint: "run", Kind: "job", Payload: []byte("{}"), ParentRequestID: parent.ID, ParentCallIndex: 0, ChildIntentDigest: childDigest("c"), ChildTargetDigest: childDigest("d")}
	child, _, problem := st.SubmitChild(call, 1, childDigest("1"), "private-boot", nil)
	fatal(t, problem)
	if child.Worker != "wanted" || child.RequestedRental != "wanted" {
		t.Fatalf("child escaped affinity: %+v", child)
	}
	_, problem = st.BlockRetainedWork(child.ID, "proof", "pause child")
	fatal(t, problem)
	retried, _, problem := st.Submit(records.Request{ID: "retry", IdemKey: "retry", BodyDigest: childDigest("e"), Package: "local/op", Entrypoint: "run", Kind: "job", Payload: []byte("{}"), RetainWork: true, RetryOf: child.ID})
	fatal(t, problem)
	if retried.Worker != "wanted" || retried.RequestedRental != "wanted" {
		t.Fatalf("retry escaped affinity: %+v", retried)
	}
	if changed, problem := st.UnpinRentalWork(retried.ID, "wanted"); problem != nil || changed {
		t.Fatalf("retry became automatic: %v %v", changed, problem)
	}
	call.ID, call.IdemKey, call.ParentCallIndex, call.RequestedRental = "wrong-child", "wrong-child", 1, "other"
	if _, _, problem := st.SubmitChild(call, 1, childDigest("1"), "private-boot", nil); problem == nil {
		t.Fatal("child replaced parent affinity")
	}
}

func TestRequestedRentalIsSubmissionIdentity(t *testing.T) {
	o := hostOwner(t, "requested-rental-identity")
	for _, kind := range []string{"serving", "job"} {
		sub := orchestrator.Submission{Kind: kind, Package: "proof/model", Entrypoint: "run", Payload: []byte("{}"), IdemKey: kind, Rental: true, RentalRequired: true, RequestedRental: "wanted"}
		first, fresh, problem := o.c.RecordSubmission(sub)
		fatal(t, problem)
		if !fresh {
			t.Fatal("first affinity submission replayed")
		}
		replay, fresh, problem := o.c.RecordSubmission(sub)
		fatal(t, problem)
		if fresh || replay.ID != first.ID {
			t.Fatal("same affinity did not replay")
		}
		sub.RequestedRental = "other"
		if _, _, problem := o.c.RecordSubmission(sub); problem == nil {
			t.Fatal("idempotency key changed rental affinity")
		}
	}
}

func TestNamedRentalReplayAfterEndUsesExistingDaemonRequest(t *testing.T) {
	for _, kind := range []string{"serving", "job"} {
		t.Run(kind, func(t *testing.T) {
			h := newLadderHub(t)
			h.bind(goodLadder())
			root, _ := placementRoot(t, h, "balanced")
			cert := filepath.Join(root, "fixture.pem")
			must(t, os.WriteFile(cert, []byte("fixture"), 0600))
			st, problem := records.Open(filepath.Join(root, "creator.sqlite"))
			fatal(t, problem)
			id := "pr-replayrental"
			fatal(t, st.RecordRental(records.Rental{ID: id, MachineName: "isao", SKU: "h100-80", AcceleratorModel: h100SXM, AcceleratorCount: 1, HourlyRateUSDMicros: 2490000, State: "ready", Address: "127.0.0.1:1", CertPath: cert, Hub: h.server.URL}))
			h.addReady(id, "isao", h100SXM, 2490000)
			st.Close()
			installedHere(t, root, h.server.URL, ladderPackage, "1.0.0")
			startDaemonProcess(t, root)
			args := []string{"run", ladderPackage + "/generate", "steps=1", "--rental=isao", "--json", "--idempotency-key=ended-replay"}
			if kind == "job" {
				args = []string{"run", ladderPackage + "/lane", "--model.pruned=proof/minimax@1.0.0-rc.1/fp8-adaln-pruned", "--rental=isao", "--json", "--idempotency-key=ended-replay"}
			}
			_, firstOut := runCozy(t, root, args...)
			st, problem = records.Open(filepath.Join(root, "creator.sqlite"))
			fatal(t, problem)
			defer st.Close()
			first, problem := st.RequestByIdempotencyKey("ended-replay")
			fatal(t, problem)
			if first == nil {
				t.Fatalf("initial named request absent: %s", firstOut)
			}
			_, _ = runCozy(t, root, "run", "cancel", first.ID, "--json")
			before, problem := st.RequestRow(first.ID)
			fatal(t, problem)
			if before.State != "canceled" && before.State != "failed" {
				t.Fatalf("request has not settled: %s", before.State)
			}
			h.mu.Lock()
			delete(h.rentals, id)
			h.mu.Unlock()
			_, problem = st.ForgetRental(id)
			fatal(t, problem)
			_, replayOut := runCozy(t, root, args...)
			after, problem := st.RequestByIdempotencyKey("ended-replay")
			fatal(t, problem)
			if after == nil || after.ID != before.ID || after.State != before.State || after.Ordinal != before.Ordinal {
				t.Fatalf("replay reactivated or replaced request: before=%+v after=%+v output=%s", before, after, replayOut)
			}
			if strings.Contains(replayOut, "rental.selection_unavailable") || strings.Contains(replayOut, "rental.dependency_mismatch") {
				t.Fatalf("replay consulted ended rental: %s", replayOut)
			}
			if asks := h.postedSKUs(); len(asks) != 0 {
				t.Fatalf("replay purchased capacity: %v", asks)
			}
		})
	}
}

func TestRentalInventoryRefusesAnUnreadablePython(t *testing.T) {
	inventory := &pb.ImageInventory{Interpreters: []*pb.PythonInterpreter{{Version: "broken", Abi: "cp312"}}}
	if _, why := launch.InventoryPython(inventory, ">=3.12", ""); why == "" {
		t.Fatal("an unreadable Python version was accepted")
	}
	if _, why := launch.InventoryPython(&pb.ImageInventory{Python: "3.12.11"}, ">=3.12", ""); !strings.Contains(why, "no available Python executor") {
		t.Fatal(why)
	}
}
