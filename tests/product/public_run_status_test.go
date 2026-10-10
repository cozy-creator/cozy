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

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
)

// A retained job that fails before it starts is failed: listed, filtered and watched as a
// failure, owing nothing, with --retry offered while its machine can still take one.
func TestPublicRunStatusShowsFailedRetainedWork(t *testing.T) {
	o := hostOwner(t, "public-run-status")
	manual := recordPrivateTransaction(t, o.store, "manual", "")
	_, problem := o.store.FailQueuedRequest(manual.ID, retainedFailure("dependency.missing", "fix the dependency before retrying"))
	fatal(t, problem)
	unavailable := recordPrivateTransaction(t, o.store, "no-custody", "unavailable-rental")
	_, problem = o.store.FailQueuedRequest(unavailable.ID, retainedFailure("dependency.missing", "the selected machine is unavailable"))
	fatal(t, problem)
	paused := recordPrivateTransaction(t, o.store, "paused", "")
	_, problem = o.store.RequestPause(paused.ID, "test user")
	fatal(t, problem)
	_, problem = o.store.CompleteRequestPause(paused.ID)
	fatal(t, problem)
	refused, _, problem := o.store.Submit(records.Request{ID: "refused-at-the-door", IdemKey: "refused-at-the-door", Kind: "job", Package: "local/private-proof", Entrypoint: "prepare", Payload: []byte(`{}`), BodyDigest: childDigest("1"), RetainWork: true, MachineExecutionObserver: true})
	fatal(t, problem)
	fatal(t, o.store.LinkMachineExecution(refused.ID, "local"))
	_, problem = o.store.MarkRunV1Sent(refused.ID)
	fatal(t, problem)
	_, problem = o.store.FailQueuedRequest(refused.ID, retainedFailure("machine.upgrade_required", "the machine refused the run"))
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
	if len(all) != 4 || document["failed"] != float64(3) || document["queued"] != nil || document["paused"] != float64(1) {
		t.Fatalf("inconsistent public census: %+v", document)
	}
	_, failed := read("failed")
	_, queued := read("queued")
	_, stopped := read("paused")
	if len(failed) != 3 || len(queued) != 0 || len(stopped) != 1 || stopped[0]["status"] != "paused" {
		t.Fatal("public filters disagree with projected rows")
	}
	for _, entry := range failed {
		wantRetry := entry["number"] == float64(manual.Number)
		if entry["status"] != "failed" || (entry["retry_available"] == true) != wantRetry || entry["retaining"] == true {
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
		if code == 0 || json.Unmarshal([]byte(out), &result) != nil || result.Error.Code != "failed" {
			t.Fatalf("a failed retained job was not a failure [%d]: %s", code, out)
		}
		if strings.Contains(out, "--retry") != item.retry {
			t.Fatalf("incorrect retained retry hint: %s", out)
		}
		row, problem := o.store.RequestRow(item.row.ID)
		fatal(t, problem)
		if row.State != "failed" {
			t.Fatalf("watching changed the run: %+v", row)
		}
	}
	row, problem := o.store.RequestRow(refused.ID)
	fatal(t, problem)
	if o.store.RetainedRetryAvailable(*row) {
		t.Fatal("a sent run advertised retry")
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
					fmt.Fprint(w, "id: 3\ndata: "+`{"type":"request.blocked","request_id":"legacy-stop","sequence_number":3,"at":"2026-09-26T01:00:01Z","payload":{"status":"blocked","error_type":"dependency.missing","error":"repair input"}}`+"\n\n")
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
