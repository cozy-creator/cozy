package producttest

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func TestPublicRunStatusProjectsStoppedFailuresAndPendingAcceptanceConsistently(t *testing.T) {
	o := hostOwner(t, "public-run-status")
	manual := recordPrivateTransaction(t, o.store, "manual", "")
	_, problem := o.store.BlockRetainedWork(manual.ID, "dependency.missing", "fix the dependency before retrying")
	fatal(t, problem)
	unavailable := recordPrivateTransaction(t, o.store, "no-custody", "unavailable-rental")
	_, problem = o.store.BlockRetainedWork(unavailable.ID, "dependency.missing", "the selected machine is unavailable")
	fatal(t, problem)
	paused := recordPrivateTransaction(t, o.store, "paused", "")
	_, problem = o.store.RequestPause(paused.ID, "test user")
	fatal(t, problem)
	_, problem = o.store.CompleteRequestPause(paused.ID)
	fatal(t, problem)
	pending, _, problem := o.store.Submit(records.Request{ID: "pending-acceptance", IdemKey: "pending-acceptance", Kind: "job", Package: "local/private-proof", Entrypoint: "prepare", Payload: []byte(`{}`), BodyDigest: childDigest("1"), RetainWork: true, MachineExecutionObserver: true})
	fatal(t, problem)
	fatal(t, o.store.LinkMachineExecution(pending.ID, "local"))
	capture, spec := []byte(`{"capture":"frozen"}`), []byte(`{"invocation":"frozen"}`)
	fatal(t, o.store.RecordMachineSubmission(pending.ID, &pb.MachineExecutionSubmit{SubmissionId: pending.IdemKey, ExpectedExecutionWorkspaceId: "workspace", CaptureCanonicalBytes: capture, CaptureDigest: canonical.Digest(capture), Offer: &pb.AttemptOffer{RequestId: pending.ID, AttemptOrdinal: 1, InvocationSpecCanonicalBytes: spec, InvocationSpecDigest: canonical.Digest(spec)}}))
	_, problem = o.store.BlockRetainedWork(pending.ID, "connection.lost", "acceptance receipt has not arrived")
	fatal(t, problem)
	defer publicationControlAPI(t, o)()

	read := func(state string) (map[string]any, []map[string]any) {
		t.Helper()
		args := []string{"run", "list", "--json"}
		if state != "" {
			args = append(args, "--state", state)
		}
		code, out := runCozy(t, o.root, args...)
		var document map[string]any
		if code != 0 || json.Unmarshal([]byte(out), &document) != nil {
			t.Fatalf("list %s [%d]: %s", state, code, out)
		}
		var result []map[string]any
		for _, value := range document["invocations"].([]any) {
			result = append(result, value.(map[string]any))
		}
		return document, result
	}
	document, all := read("")
	if _, exists := document["blocked"]; exists {
		t.Fatalf("public blocked category: %+v", document)
	}
	if len(all) != 4 || document["failed"] != float64(2) || document["queued"] != float64(1) || document["paused"] != float64(1) {
		t.Fatalf("inconsistent public census: %+v", document)
	}
	_, failed := read("failed")
	_, queued := read("queued")
	_, stopped := read("paused")
	if len(failed) != 2 || len(queued) != 1 || queued[0]["status"] != "queued" || len(stopped) != 1 || stopped[0]["status"] != "paused" {
		t.Fatal("public filters disagree with projected rows")
	}

	for _, entry := range failed {
		wantRetry := entry["number"] == float64(manual.Number)
		if entry["status"] != "failed" || (entry["retry_available"] == true) != wantRetry || entry["retaining"] != true {
			t.Fatalf("failed list lost recovery facts: %+v", entry)
		}
	}

	for _, item := range []struct {
		row   records.Request
		retry bool
	}{{manual, true}, {unavailable, false}} {
		code, out := runCozy(t, o.root, "run", "watch", strconv.FormatInt(item.row.Number, 10), "--json")
		var result struct {
			Error struct {
				Code, Message string
				Next          []string
			}
		}
		if code == 0 || json.Unmarshal([]byte(out), &result) != nil || result.Error.Code != "failed" || strings.Contains(result.Error.Message, "ended blocked") {
			t.Fatalf("manual stop was not a failure [%d]: %s", code, out)
		}
		if strings.Contains(out, "--retry") != item.retry {
			t.Fatalf("incorrect retained retry hint: %s", out)
		}
		row, problem := o.store.RequestRow(item.row.ID)
		fatal(t, problem)
		if row.State != "blocked" || !row.RetainWork {
			t.Fatal("presentation changed scheduling/custody")
		}
	}
	before, problem := o.store.MachineExecution(pending.ID)
	fatal(t, problem)
	if len(before.Submission) == 0 || len(before.Receipt) != 0 || len(before.PendingControl) != 0 {
		t.Fatal("public observation invented acceptance or a control operation")
	}
	row, problem := o.store.RequestRow(pending.ID)
	fatal(t, problem)
	if o.store.RetainedRetryAvailable(*row) {
		t.Fatal("ambiguous acceptance advertised retry")
	}
	// Public failure remains cancellable when it still holds private work.
	code, out := runCozy(t, o.root, "run", "cancel", strconv.FormatInt(manual.Number, 10), "--json")
	var canceled struct{ Changed bool }
	if code != 0 || json.Unmarshal([]byte(out), &canceled) != nil || !canceled.Changed {
		t.Fatalf("failed retained work could not be released [%d]: %s", code, out)
	}
	row, problem = o.store.RequestRow(manual.ID)
	fatal(t, problem)
	if row.State != "canceled" {
		t.Fatalf("retained work cancellation did not reach store: %s", row.State)
	}
}

func TestPublicRunStatusFreezesStoppedExecutionAtOriginalEvent(t *testing.T) {
	o := hostOwner(t, "public-stopped-duration")
	request, original := retainedLostObserver(t, o.store, "stopped-duration")
	ended, err := time.Parse(time.RFC3339Nano, original.At)
	must(t, err)
	defer publicationControlAPI(t, o)()
	for range 2 {
		if got, want := listedMachineExecutionMS(t, o.root, request.ID), ended.UnixMilli()-1000; got != want {
			t.Fatalf("manual stop execution_ms=%d, want original event duration %d", got, want)
		}
	}
	link, problem := o.store.MachineExecution(request.ID)
	fatal(t, problem)
	if len(link.Outcome) != 0 {
		t.Fatal("public failure invented a Runtime terminal outcome")
	}
}

func TestPublicRunStatusWatchAcceptsOlderDaemonWithoutStopIdentity(t *testing.T) {
	for _, kind := range []string{"job", "invocation"} {
		t.Run(kind, func(t *testing.T) {
			var streams atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/" {
					return
				}
				if strings.HasSuffix(r.URL.Path, "/events") {
					streams.Add(1)
					w.Header().Set("Content-Type", "text/event-stream")
					fmt.Fprint(w, "id: 3\ndata: "+`{"type":"request.blocked","request_id":"legacy-stop","event_id":3,"at":"2026-09-26T01:00:01Z","payload":{"status":"blocked","error_type":"dependency.missing","error":"repair input"}}`+"\n\n")
					return
				}
				// Deliberately omit retry_available and stopped_event_id, as old releases do.
				_ = json.NewEncoder(w).Encode(map[string]any{"number": 1, "kind": kind, "request_id": "legacy-stop", "job_id": "legacy-stop", "status": "blocked", "created_at": "2026-09-26T01:00:00Z", "error_type": "dependency.missing", "error": "repair input"})
			}))
			defer server.Close()
			layout, problem := home.Open(t.TempDir())
			fatal(t, problem)
			held, problem := daemon.Hold(layout, strings.TrimPrefix(server.URL, "http://"), "")
			fatal(t, problem)
			defer held.Release()
			_, problem = api.Mint(layout)
			fatal(t, problem)
			code, out := runCozy(t, layout.Root, "run", "watch", "1", "--json")
			var result struct{ Error struct{ Code string } }
			if code == 0 || json.Unmarshal([]byte(out), &result) != nil || result.Error.Code != "failed" || strings.Contains(out, "--retry") || strings.Contains(out, "ended blocked") || streams.Load() != 1 {
				t.Fatalf("older daemon manual stop was not tolerated [%d] streams=%d: %s", code, streams.Load(), out)
			}
		})
	}
}
