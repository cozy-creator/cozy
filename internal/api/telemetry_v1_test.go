package api

import (
	"bufio"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/records"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

func TestMachineTelemetryIsLiveAtSubscribeAndCannotAdvanceTheDurableSSECursor(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "records.sqlite"))
	if problem != nil {
		t.Fatal(problem)
	}
	defer store.Close()
	_, _, problem = store.Submit(records.Request{ID: "run", IdemKey: "run", Package: "local/test", Entrypoint: "main", Kind: "call",
		Payload: []byte(`{}`), BodyDigest: "sha256:" + strings.Repeat("1", 64), MachineExecutionObserver: true})
	if problem != nil {
		t.Fatal(problem)
	}
	if problem = store.LinkMachineExecution("run", "machine"); problem != nil {
		t.Fatal(problem)
	}
	if problem = store.AcceptRunV1("run", &v1.RunState{Id: "run", Number: 1, State: "running", Attempt: 1}); problem != nil {
		t.Fatal(problem)
	}
	sample := &v1.RunEvent{Sequence: 40, Event: &v1.RunEvent_Progress{Progress: &v1.Progress{Stage: "denoise", Fraction: .4}}}
	if problem = store.ObserveRunV1("run", sample, nil); problem != nil {
		t.Fatal(problem)
	}
	owner := &Server{store: store}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { owner.stream(w, r, "") }))
	defer server.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, server.URL, nil)
	response, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	reader := bufio.NewReader(response.Body)
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			t.Fatal(err)
		}
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		var event Envelope
		if err := json.Unmarshal([]byte(strings.TrimSpace(strings.TrimPrefix(line, "data: "))), &event); err != nil {
			t.Fatal(err)
		}
		if event.Type != "machine.progress" {
			continue
		}
		if event.SequenceNumber != 0 || event.RequestID != "run" || event.Payload["live"] != true {
			t.Fatalf("live sample became a durable event: %v", event)
		}
		payload, ok := event.Payload["payload"].(map[string]any)
		if !ok || payload["stage"] != "denoise" || payload["overall_fraction"] != .4 {
			t.Fatalf("latest sample missing: %v", event)
		}
		break
	}
	link, problem := store.MachineExecution("run")
	if problem != nil || link.RemoteCursor != 0 {
		t.Fatalf("live SSE skipped durable output events: %v %v", link, problem)
	}
}
