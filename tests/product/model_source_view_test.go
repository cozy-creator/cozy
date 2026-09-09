package producttest

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
)

// TestAFailedSourceMemberReachesTheClientAndTheCLI is the half of run 205 that would have
// let a human answer the question in a minute.
//
// cozy#305 carried `safe_code`/`safe_detail` from the pod into the durable row and into the
// live progress frame. Nothing read them back: `grep -rn safe_code internal/api internal/cli`
// was empty, and the job document summed the source rows into three numbers. So an operator
// on `cozy run watch` saw a percentage for 2h55m while the row beside it named the member
// that had refused.
//
// The arm asserts the SURFACES — the HTTP document and the CLI's own output — never the row.
// The row already had it, and that is the whole defect.
func TestAFailedSourceMemberReachesTheClientAndTheCLI(t *testing.T) {
	const (
		requestID = "job-source-verdict-in-view"
		code      = "source_fetch_refused"
		detail    = "the origin answered 401 for b.safetensors and will not answer another way"
	)
	root := t.TempDir()
	daemon := startDaemonProcess(t, root)

	// The request is recorded through the store the daemon is serving. Dispatch is not the
	// subject here — cl-133's arm drives the whole wire — and a row the daemon's queue
	// never saw stays exactly where it is put.
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()

	intent := &records.ModelTransferIntent{
		Kind: "model-upload", Destination: "paul/minimax-h3",
		Source:          "hf://MiniMaxAI/MiniMax-H3@" + strings.Repeat("4", 40),
		SourceSelection: "sha256:" + strings.Repeat("2", 64),
		SourceFiles: []records.ModelTransferSourceFile{
			{Member: "a.safetensors", SHA256: strings.Repeat("a", 64), Length: 110_331_470_098},
			{Member: "b.safetensors", SHA256: strings.Repeat("b", 64), Length: 100_000_000_000},
		},
		Outputs: []records.ModelTransferOutput{{Name: "full"}},
	}
	_, created, problem := store.Submit(records.Request{
		ID: requestID, IdemKey: requestID, BodyDigest: "sha256:" + strings.Repeat("c", 64),
		Package: "paul/minimax-h3-tools", Entrypoint: "four-lane", State: "queued",
		Kind: "job", Payload: []byte("{}"), Outputs: "[]", WeightsOutputs: "[]",
		ModelTransfer: intent,
	})
	fatal(t, problem)
	if !created {
		t.Fatal("the transfer request was not created; the arm proves nothing")
	}
	// 1 of 2 verified, and the other one refused: run 205's shape.
	fatal(t, store.RecordModelTransferSourceStatus(records.ModelTransferSourceStatus{
		RequestID: requestID, Member: "a.safetensors",
		ObjectID: "sha256:" + strings.Repeat("a", 64), Length: 110_331_470_098,
		CapabilityRevision: 1, State: "verified", Transferred: 110_331_470_098}))
	fatal(t, store.RecordModelTransferSourceStatus(records.ModelTransferSourceStatus{
		RequestID: requestID, Member: "b.safetensors",
		ObjectID: "sha256:" + strings.Repeat("b", 64), Length: 100_000_000_000,
		CapabilityRevision: 1, State: "failed", SafeCode: code, SafeDetail: detail}))

	// SURFACE ONE, while the request is still live. The rows used to be read only while the
	// transfer was `materializing` — only while there was nothing to explain.
	answer := daemon.call(t, "GET", "/v1/local/jobs/"+requestID, nil)
	if answer.Status != 200 {
		t.Fatalf("the job document refused: %s", answer.brief())
	}
	var state api.JobState
	must(t, json.Unmarshal(answer.Body, &state))
	if len(state.ModelSources) != 1 {
		t.Fatalf("the job document carries %d source row(s), want the one that did not "+
			"verify: %s", len(state.ModelSources), answer.brief())
	}
	source := state.ModelSources[0]
	if source.Member != "b.safetensors" || source.State != "failed" ||
		source.SafeCode != code || source.SafeDetail != detail {
		t.Fatalf("the job document says %+v; want b.safetensors failed carrying the pod's "+
			"own %q: %q", source, code, detail)
	}

	// SURFACE TWO, once the transfer has settled: what a person actually runs.
	fatal(t, store.FailModelTransfer(requestID, code, detail))
	settled, problem := store.FailModelTransferRequest(requestID, code, detail,
		map[string]any{"status": "FAILED", "cause": code, "error_type": code,
			"error": detail, "outputs": []any{}, "requeuing": false})
	fatal(t, problem)
	if !settled {
		t.Fatal("the transfer request did not settle; the CLI arm has nothing to watch")
	}
	_, out, errOut := runCozyStreams(t, root, "run", "watch", requestID)
	said := out + errOut
	if !strings.Contains(said, "b.safetensors") || !strings.Contains(said, code) {
		t.Fatalf("`cozy run watch` never named the member that refused or why:\n%s", said)
	}
	// And a verified member is not noise on the way to it.
	if strings.Contains(said, "a.safetensors") {
		t.Fatalf("`cozy run watch` listed a member that verified:\n%s", said)
	}
}

func TestCompletedSourcePublicationSuppressesStalePendingRows(t *testing.T) {
	root := t.TempDir()
	daemon := startDaemonProcess(t, root)
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	id := "job-native-source-complete"
	intent := &records.ModelTransferIntent{Kind: "model-upload", Destination: "paul/minimax-h3", Source: "hf://proof/pdd@" + strings.Repeat("4", 40), SourceSelection: childDigest("2"), SourceFiles: []records.ModelTransferSourceFile{{Member: "pdd.safetensors", SHA256: strings.Repeat("a", 64), Length: 1372450680}}, Outputs: []records.ModelTransferOutput{{Name: "adapter"}}}
	_, _, problem = store.Submit(records.Request{ID: id, IdemKey: id, BodyDigest: childDigest("c"), Package: "local/pdd", Entrypoint: "main", Kind: "job", Payload: []byte("{}"), ModelTransfer: intent})
	fatal(t, problem)
	read := func() api.JobState {
		answer := daemon.call(t, "GET", "/v1/local/jobs/"+id, nil)
		if answer.Status != 200 {
			t.Fatal(answer.brief())
		}
		var state api.JobState
		must(t, json.Unmarshal(answer.Body, &state))
		return state
	}
	if sources := read().ModelSources; len(sources) != 1 || sources[0].State != "pending" || sources[0].Transferred != 0 {
		t.Fatalf("pending source was hidden: %+v", sources)
	}
	fatal(t, store.CompleteModelTransferMaterialization(id, []records.ModelRef{{Slot: "adapter", Manifest: childDigest("d"), ManifestLength: 163}}, ""))
	fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "source-worker", Package: "local/pdd", WorkerID: "source-worker", Devices: []string{"cpu"}}))
	ordinal, problem := store.Dispatch(records.Attempt{RequestID: id, InstanceID: "source-worker", SessionID: "source-session", InvocationDigest: childDigest("e"), InvocationCanonical: []byte("{}")})
	fatal(t, problem)
	fatal(t, store.RecordModelTransferWeights(records.ModelTransferWeights{RequestID: id, OutputSlot: "adapter", ManifestID: childDigest("d"), ManifestLength: 163, Attempt: ordinal, InvocationDigest: childDigest("e"), TransactionID: childDigest("f"), ReceiptDigest: childDigest("1")}))
	fatal(t, store.CompleteModelTransferOutput(id, ordinal, "adapter", "published-adapter"))
	fatal(t, store.BeginModelTransferFinalization(id))
	if sources := read().ModelSources; len(sources) != 1 {
		t.Fatal("publication still pending but source report disappeared")
	}
	fatal(t, store.CompleteModelTransfer(id, map[string]string{"adapter": childDigest("d")}))
	state := read()
	if len(state.ModelSources) != 0 || state.ModelOutputs["adapter"] != childDigest("d") {
		t.Fatalf("completed source publication still pending: %+v", state)
	}
	statuses, problem := store.ModelTransferSourceStatuses(id)
	fatal(t, problem)
	if len(statuses) != 1 || statuses[0].State != "pending" || statuses[0].Transferred != 0 {
		t.Fatal("readback fabricated raw transfer observations")
	}
	// The parent has not been marked successful: the source-publication authority,
	// rather than an unrelated parent terminal, determines this projection.
	row, problem := store.RequestRow(id)
	fatal(t, problem)
	if row.State == "succeeded" {
		t.Fatal("fixture accidentally depended on parent success")
	}
}
