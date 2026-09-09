package producttest

import (
	"database/sql"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/exit"
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
		mux.HandleFunc("GET /v1/rentals/{id}/image-inventory", func(w http.ResponseWriter, r *http.Request) {
			inventoryReads.Add(1)
			version := "0.15.0"
			if r.PathValue("id") == old {
				version = "0.13.0"
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"image_inventory": map[string]any{
				"format": "tensorhub.image_inventory/1", "profile": "python3.12-cpu-linux-x86", "python": "3.12.12",
				"distributions": []map[string]string{{"name": runtimeDistribution, "version": version}},
			}})
		})
	})
	st, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	for id, name := range map[string]string{current: "isao", old: "giriko"} {
		fatal(t, st.RecordRental(records.Rental{ID: id, MachineName: name, AcceleratorModel: "CPU", AcceleratorCount: 1, State: "ready", HourlyRateUSDMicros: 100000, Address: "127.0.0.1:1", CertPath: "unused", Hub: "fixture"}))
	}
	st.Close()
	args := []string{"run", "proof/quantize/quantize", "steps=7", "model.dits=proof/source@1.0.0/bf16", "model.shared=proof/source@1.0.0/bf16", "--dry-run", "--json", "--full"}
	for _, name := range []string{"isao", current} {
		code, out := runCozy(t, root, append(append([]string{}, args...), "--rental="+name)...)
		if code != 0 || !strings.Contains(out, `"requested_rental":"`+current+`"`) {
			t.Fatalf("named rental %s: %d %s", name, code, out)
		}
	}
	code, out := runCozy(t, root, append(append([]string{}, args...), "--rental=giriko")...)
	if code == 0 || !strings.Contains(out, "rental.dependency_mismatch") || !strings.Contains(out, "0.13.0") {
		t.Fatalf("wrong runtime admitted: %d %s", code, out)
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
	if _, err := os.Stat(filepath.Join(root, "daemon.lock")); !os.IsNotExist(err) {
		t.Fatal("dryrun/invalid selection started a daemon")
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

func TestRequestedRentalLossSettlesWithoutFallback(t *testing.T) {
	var purchases atomic.Int32
	o := hostOwner(t, "explicit-rental-loss", func(opt *orchestrator.Options) {
		opt.AcquireManagedRental = func(records.Request) (orchestrator.PlacementDecision, string, *exit.Error) {
			purchases.Add(1)
			return orchestrator.PlacementDecision{}, "", nil
		}
	})
	for _, id := range []string{"queued", "running"} {
		_, _, problem := o.store.Submit(records.Request{ID: id, IdemKey: id, BodyDigest: "sha256:" + strings.Repeat("1", 64), Package: "proof/model", Entrypoint: "run", Payload: []byte("{}"), Rental: true, RentalRequired: true, Worker: "gone", RequestedRental: "gone"})
		fatal(t, problem)
	}
	fatal(t, o.store.SpawnWorker(records.WorkerProcess{InstanceID: "lost", Package: "proof/model", WorkerID: "lost", Devices: []string{"cpu"}}))
	ordinal, problem := o.store.Dispatch(records.Attempt{RequestID: "running", SessionID: "session", InstanceID: "lost", InvocationDigest: "sha256:" + strings.Repeat("2", 64), InvocationCanonical: []byte("{}")})
	fatal(t, problem)
	fatal(t, o.store.OfferDispatch("running", ordinal, "session"))
	fatal(t, o.store.Accepted("running", ordinal, "session"))
	o.c.RecoverLostWork()
	for _, id := range []string{"queued", "running"} {
		row, problem := o.store.RequestRow(id)
		fatal(t, problem)
		if row.State != "failed" || row.Worker != "gone" || row.RequestedRental != "gone" {
			t.Fatalf("fixed rental loss replanned %s: %+v", id, row)
		}
	}
	events, problem := o.store.EventsAfter("running", 0, 100)
	fatal(t, problem)
	terminal := false
	for _, event := range events {
		if event.Type == "request.failed" && event.Payload["error_type"] == "rental.selected_lost" && event.Payload["requeuing"] == false {
			terminal = true
		}
	}
	if !terminal {
		t.Fatal("lost active attempt has no terminal event")
	}
	if purchases.Load() != 0 {
		t.Fatal("loss bought replacement")
	}
}

func TestSchema36MigrationKeepsAutomaticAssignments(t *testing.T) {
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	st, problem := records.Open(path)
	fatal(t, problem)
	_, _, problem = st.Submit(records.Request{ID: "existing", IdemKey: "existing", BodyDigest: "sha256:" + strings.Repeat("3", 64), Package: "proof/model", Entrypoint: "run", Payload: []byte("{}"), Rental: true, Worker: "prior"})
	fatal(t, problem)
	st.Close()
	db, err := sql.Open("sqlite", path)
	must(t, err)
	var create string
	must(t, db.QueryRow(`SELECT sql FROM sqlite_master WHERE name='requests'`).Scan(&create))
	create = strings.Replace(create, ",\n  requested_rental TEXT NOT NULL DEFAULT ''", "", 1)
	_, err = db.Exec(`ALTER TABLE requests DROP COLUMN requested_rental; PRAGMA user_version=36`)
	must(t, err)
	_, err = db.Exec(`PRAGMA writable_schema=ON; UPDATE sqlite_master SET sql=? WHERE name='requests'; PRAGMA writable_schema=OFF`, create)
	must(t, err)
	db.Close()
	if prior, problem := records.Open(path); problem == nil {
		prior.Close()
		t.Fatal("reader silently migrated shared state")
	}
	st, problem = records.OpenForDaemon(path, "")
	fatal(t, problem)
	defer st.Close()
	row, problem := st.RequestRow("existing")
	fatal(t, problem)
	if row.Worker != "prior" || row.RequestedRental != "" {
		t.Fatalf("migration converted automatic pin into affinity: %+v", row)
	}
}

func TestRentPrimaryAndHiddenCreationAliasShareBehavior(t *testing.T) {
	root, _, _, _, _ := runModelCatalog(t)
	for _, args := range [][]string{{"rent", "--json"}, {"rental", "new", "--json"}} {
		code, out := runCozy(t, root, args...)
		if code != 0 || !strings.Contains(out, "cpu") {
			t.Fatalf("catalog %v: %d %s", args, code, out)
		}
	}
	code, out := runCozy(t, root, "rental", "--help")
	if code != 0 || strings.Contains(out, "rental new") {
		t.Fatalf("compatibility alias advertised: %d %s", code, out)
	}
}

func TestAutomaticAndNamedRentalChoicesRespectExistingImage(t *testing.T) {
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
	startDaemonProcess(t, root)
	for _, arm := range []struct{ key, flag string }{{"automatic-runtime", "--rental-only"}, {"named-runtime", "--rental=isao"}} {
		_, out := runCozy(t, root, "run", ladderPackage+"/generate", "steps=1", arm.flag, "--json", "--idempotency-key", arm.key)
		st, problem = records.Open(filepath.Join(root, "creator.sqlite"))
		fatal(t, problem)
		row, problem := st.RequestByIdempotencyKey(arm.key)
		fatal(t, problem)
		if row == nil || row.Worker != "pr-newruntime" {
			t.Fatalf("%s chose wrong rental: %+v %s", arm.key, row, out)
		}
		if arm.key == "named-runtime" && row.RequestedRental != "pr-newruntime" {
			t.Fatal("explicit request lost durable affinity")
		}
		st.Close()
	}
	if asks := h.postedSKUs(); len(asks) != 0 {
		t.Fatalf("compatible attached rental lost to purchase: %v", asks)
	}
}

func TestRentalInventoryAllowsPackageOwnedVersions(t *testing.T) {
	inventory := &pb.ImageInventory{Python: "3.12.12", Distributions: []*pb.ImageDistribution{
		{Distribution: runtimeDistribution, Version: "0.13.0"}, {Distribution: "diffusers", Version: "0.39.0"}, {Distribution: "tokenizers", Version: "0.22.0"},
	}}
	if why := launch.InventoryMismatch(inventory, []string{"diffusers==0.40.0", "tokenizers==0.23.1"}, ">=3.12,<3.13"); why != "" {
		t.Fatal(why)
	}
	for _, requirement := range []string{"cozy-runtime>=0.15.0", "Cozy_Runtime[media]>=0.15.0", "  cozy..runtime >=0.15.0"} {
		if why := launch.InventoryMismatch(inventory, []string{requirement}, ""); !strings.Contains(why, "cozy-runtime 0.13.0") {
			t.Fatalf("%s mismatch: %s", requirement, why)
		}
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
