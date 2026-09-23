package hub

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/secret"
)

func testFinalizeClient(t *testing.T, handler http.Handler) *Client {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	return New(config.Config{HubURL: server.URL, HubToken: secret.New("test-token")}, "publish-test")
}

func checkpointFixture() CheckpointPublication {
	return CheckpointPublication{
		PublishID: "op", CheckpointID: "sha256:" + "a" + "000000000000000000000000000000000000000000000000000000000000000",
		Manifest: ManifestRef{SHA256: "b" + "000000000000000000000000000000000000000000000000000000000000000", Length: 10},
		Objects:  2, Bytes: 20, State: "checkpointed",
	}
}

func TestFinalizePublicationPollsAsyncResult(t *testing.T) {
	oldInterval := modelFinalizePollInterval
	modelFinalizePollInterval = time.Millisecond
	t.Cleanup(func() { modelFinalizePollInterval = oldInterval })

	var statusCalls atomic.Int32
	checkpoint := checkpointFixture()
	client := testFinalizeClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test-token" {
			t.Fatalf("authorization = %q", r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "application/json")
		switch r.Method {
		case http.MethodPost:
			if r.URL.Path != "/v1/models/acme/demo/publications/op/finalize" {
				t.Fatalf("finalize path = %q", r.URL.Path)
			}
			w.WriteHeader(http.StatusAccepted)
			_, _ = w.Write([]byte(`{"operation":"op","state":"queued","status_url":"/v1/models/acme/demo/publications/op/finalization"}`))
		case http.MethodGet:
			if r.URL.Path != "/v1/models/acme/demo/publications/op/finalization" {
				t.Fatalf("status path = %q", r.URL.Path)
			}
			if statusCalls.Add(1) == 1 {
				_, _ = w.Write([]byte(`{"operation":"op","state":"running","status_url":"/v1/models/acme/demo/publications/op/finalization"}`))
				return
			}
			result, err := json.Marshal(checkpoint)
			if err != nil {
				t.Fatal(err)
			}
			response := map[string]any{"operation": "op", "state": "completed", "status_url": "/v1/models/acme/demo/publications/op/finalization", "result": json.RawMessage(result)}
			_ = json.NewEncoder(w).Encode(response)
		default:
			t.Fatalf("unexpected method %s", r.Method)
		}
	}))

	got, problem := client.FinalizePublication(context.Background(), Ref{Org: "acme", Name: "demo"}, "op", FinalizePublicationRequest{ManifestID: "sha256:" + "c" + "000000000000000000000000000000000000000000000000000000000000000", ManifestLength: 10}, "model test")
	if problem != nil {
		t.Fatalf("FinalizePublication failed: %v", problem)
	}
	if got != checkpoint {
		t.Fatalf("checkpoint = %#v, want %#v", got, checkpoint)
	}
	if statusCalls.Load() != 2 {
		t.Fatalf("status calls = %d, want 2", statusCalls.Load())
	}
}

func TestFinalizePublicationReadsLegacySynchronousResult(t *testing.T) {
	checkpoint := checkpointFixture()
	client := testFinalizeClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("method = %s", r.Method)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(checkpoint)
	}))
	got, problem := client.FinalizePublication(context.Background(), Ref{Org: "acme", Name: "demo"}, "op", FinalizePublicationRequest{ManifestID: "sha256:" + "c" + "000000000000000000000000000000000000000000000000000000000000000", ManifestLength: 10}, "model test")
	if problem != nil {
		t.Fatalf("FinalizePublication failed: %v", problem)
	}
	if got != checkpoint {
		t.Fatalf("checkpoint = %#v, want %#v", got, checkpoint)
	}
}
