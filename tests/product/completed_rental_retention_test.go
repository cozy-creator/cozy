package producttest

import (
	"encoding/json"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

func TestRentalEndPreservesCompletedNativeResultHistory(t *testing.T) {
	root, origin, stand := rentalEndRoot(t, "completed-rental-retention")
	stand.publishListing()
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	for id, machine := range map[string]string{"pr-completed-rental": "otter", "pr-other-retention": "heron"} {
		stand.add(id, machine)
		fatal(t, store.RecordRental(records.Rental{ID: id, MachineName: machine,
			SKU: "cpu", AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 100_000,
			State: "ready", Hub: origin, Address: "127.0.0.1:1", CertPath: filepath.Join(root, id+".pem")}))
	}
	startDaemonProcess(t, root)
	fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "private-worker", Package: "local/native", WorkerID: "worker", Devices: []string{"cpu"}}))
	completed := func(id, rental string) records.Request {
		t.Helper()
		request, _, problem := store.Submit(records.Request{ID: id, IdemKey: id,
			BodyDigest: childDigest("3"), Package: "local/native", Entrypoint: "main", Kind: "job",
			Org: "local", PlanID: childDigest("4"), LocalPackageDigest: childDigest("5"), Payload: []byte(`{}`),
			Worker: rental, Rental: true, RetainWork: true, ChildArtifacts: true})
		fatal(t, problem)
		request = offerChildParent(t, store, request)
		output := records.ByteOutput{RequestID: id, Attempt: 1, OutputID: "report",
			Digest: childDigest("6"), Length: 100, MimeType: "application/json", ProducerRootID: childDigest("7"),
			ReceiptDigest: childDigest("8"), ManifestID: childDigest("6"), ManifestLength: 100, ContentBytes: 100}
		_, problem = store.AcceptTerminal(records.Terminal{RequestID: id, Attempt: 1,
			SessionID: "private-boot", InvocationDigest: childDigest("1"), TerminalID: "done-" + id,
			TerminalDigest: childDigest("a"), Status: "SUCCEEDED", RequestState: "finalizing",
			EventType: "request.finalizing", EventPayload: map[string]any{"status": "FINALIZING", "execution_status": "SUCCEEDED"},
			Body: []byte(`{}`), ByteOutputs: []records.ByteOutput{output}})
		fatal(t, problem)
		hold, problem := store.ReserveByteResult(id, output)
		fatal(t, problem)
		fatal(t, store.ConfirmNativeArtifact(hold.RetentionID, "private-worker", "private-boot"))
		fatal(t, store.CompleteNativeRootResult(id, 1))
		fatal(t, store.Closed(id, 1))
		return request
	}
	request := completed("req-completed-report", "pr-completed-rental")
	other := completed("req-other-report", "pr-other-retention")
	_, _, problem = store.Submit(records.Request{ID: "req-still-queued", IdemKey: "still-queued",
		BodyDigest: childDigest("b"), Package: "local/native", Entrypoint: "main", Kind: "job", Payload: []byte(`{}`),
		Worker: "pr-completed-rental", Rental: true, RetainWork: true})
	fatal(t, problem)
	before, problem := store.AttemptRow(request.ID, 1)
	fatal(t, problem)
	assets, problem := store.ReceivedByteAssets(request.ID)
	fatal(t, problem)
	if len(assets) != 1 {
		t.Fatal("completed report lacks its actual recorded native hold")
	}
	code, out := runCozy(t, root, "rental", "end", "otter", "--json")
	if code != 0 {
		t.Fatalf("ordinary rental end failed [%d]: %s", code, out)
	}
	after, problem := store.AttemptRow(request.ID, 1)
	fatal(t, problem)
	if !reflect.DeepEqual(before, after) {
		t.Fatalf("rental cleanup rewrote terminal attempt history: before=%+v after=%+v", before, after)
	}
	row, problem := store.RequestRow(request.ID)
	fatal(t, problem)
	retaining, problem := store.RequestRetaining(*row)
	fatal(t, problem)
	if row.State != "succeeded" || retaining {
		t.Fatalf("rental cleanup changed successful execution or retained removed storage: %+v retaining=%t", row, retaining)
	}
	assets, problem = store.ReceivedByteAssets(request.ID)
	fatal(t, problem)
	if len(assets) != 0 {
		t.Fatal("deleted remote report is still grantable")
	}
	holds, problem := store.NativeArtifactRetentions(request.ID)
	fatal(t, problem)
	if len(holds) != 1 || holds[0].State != "released" {
		t.Fatalf("deleted rental custody was not released: %+v", holds)
	}
	otherAssets, problem := store.ReceivedByteAssets(other.ID)
	fatal(t, problem)
	if len(otherAssets) != 1 || stand.releases("pr-other-retention") != 0 {
		t.Fatal("ending one rental discarded another rental's retained output")
	}
	queued, problem := store.RequestRow("req-still-queued")
	fatal(t, problem)
	if queued.State != "canceled" {
		t.Fatalf("unfinished work was not canceled: %s", queued.State)
	}
	code, out = runCozy(t, root, "run", "list", "--json", "--full")
	var listed struct {
		Invocations []struct {
			ID, Status string
			CanceledBy string `json:"canceled_by"`
		}
	}
	if code != 0 || json.Unmarshal([]byte(out), &listed) != nil {
		t.Fatalf("ordinary run history failed [%d]: %s", code, out)
	}
	for _, run := range listed.Invocations {
		if run.ID == request.ID && (run.Status != "completed" || run.CanceledBy != "") {
			t.Fatalf("completed report is presented as canceled: %s", out)
		}
	}
	if strings.Contains(out, `"status":"canceling"`) {
		t.Fatal("rental end left unfinished cancellation state")
	}
}
