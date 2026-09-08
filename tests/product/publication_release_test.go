package publication

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/secret"
)

// The owner client speaks its real HTTP API. This fixture controls commit/reply
// loss and competing revisions, without replacing publication logic with a fake.
type releaseServer struct {
	mu        sync.Mutex
	revision  int64
	lanes     map[string]string
	writes    int
	loseReply bool
	denied    bool
}

func (s *releaseServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.denied {
		http.Error(w, `{"error":{"code":"auth.denied","message":"revoked"}}`, http.StatusForbidden)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	if r.Method == http.MethodGet {
		lanes := []map[string]any{}
		for name, checkpoint := range s.lanes {
			lanes = append(lanes, map[string]any{"lane": name, "manifest_id": checkpoint})
		}
		releases := []map[string]any{}
		if s.revision > 0 {
			releases = append(releases, map[string]any{"release": "v1", "revision": s.revision, "lanes": lanes})
		}
		json.NewEncoder(w).Encode(map[string]any{"org": "alice", "name": "model", "releases": releases})
		return
	}
	var request struct {
		Expected int64             `json:"expected_revision"`
		Set      map[string]string `json:"set_lanes"`
	}
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		panic(err)
	}
	s.writes++
	if request.Expected != s.revision {
		http.Error(w, `{"error":{"code":"release.conflict","message":"stale revision"}}`, http.StatusConflict)
		return
	}
	for name, value := range request.Set {
		s.lanes[name] = value
	}
	s.revision++
	if s.loseReply {
		s.loseReply = false
		connection, _, err := w.(http.Hijacker).Hijack()
		if err != nil {
			panic(err)
		}
		connection.Close()
		return
	}
	lanes := []hub.ModelReleaseLane{}
	for name, id := range s.lanes {
		lanes = append(lanes, hub.ModelReleaseLane{Lane: name, CheckpointID: id})
	}
	json.NewEncoder(w).Encode(hub.ModelRelease{Release: "v1", Revision: s.revision, Lanes: lanes, Changed: true})
}

func fixture(t *testing.T) (*releaseServer, *hub.Client) {
	t.Helper()
	service := &releaseServer{revision: 1, lanes: map[string]string{"bf16": "sha256:" + strings.Repeat("1", 64)}}
	server := httptest.NewServer(service)
	t.Cleanup(server.Close)
	return service, hub.New(config.Config{HubURL: server.URL, HubToken: secret.New("test")}, "publication-test")
}

func TestLostReplyReconcilesOnlyFrozenSuccessor(t *testing.T) {
	for _, changed := range []bool{false, true} {
		t.Run(fmt.Sprint(changed), func(t *testing.T) {
			service, client := fixture(t)
			intent, problem := PrepareRelease(context.Background(), client, ReleaseRequest{Destination: "alice/model", Release: "v1", Lanes: map[string]string{"fp8": "sha256:" + strings.Repeat("2", 64)}})
			if problem != nil {
				t.Fatal(problem)
			}
			sent := false
			service.loseReply = true
			_, problem = ApplyRelease(context.Background(), client, intent, false, func() *exit.Error { sent = true; return nil })
			if problem == nil || !sent || service.revision != 2 {
				t.Fatalf("lost reply did not preserve uncertainty: %v", problem)
			}
			if changed {
				service.revision = 3
				service.lanes["bf16"] = "sha256:" + strings.Repeat("3", 64)
			}
			result, problem := ApplyRelease(context.Background(), client, intent, true, func() *exit.Error { t.Fatal("recovery attempted another mutation"); return nil })
			if changed {
				if problem == nil || problem.ErrName() != "publication.outcome_unknown" {
					t.Fatalf("changed state accepted: %+v %v", result, problem)
				}
			} else {
				if problem != nil || result.Observation != "observed_convergence" || result.Revision != 2 {
					t.Fatalf("exact successor not recovered: %+v %v", result, problem)
				}
				if result.Lanes["bf16"] == "" {
					t.Fatal("unmentioned lane lost")
				}
			}
			if service.writes != 1 {
				t.Fatalf("wrote %d times", service.writes)
			}
		})
	}
}

func TestExplicitStaleRevisionAndFreshNoop(t *testing.T) {
	service, client := fixture(t)
	desired := map[string]string{"bf16": service.lanes["bf16"]}
	stale := int64(0)
	_, problem := PrepareRelease(context.Background(), client, ReleaseRequest{Destination: "alice/model", Release: "v1", Lanes: desired, ExpectedRevision: &stale})
	if problem == nil || problem.ErrName() != "publication.release_conflict" {
		t.Fatal("explicit stale no-op admitted", problem)
	}
	intent, problem := PrepareRelease(context.Background(), client, ReleaseRequest{Destination: "alice/model", Release: "v1", Lanes: desired})
	if problem != nil {
		t.Fatal(problem)
	}
	result, problem := ApplyRelease(context.Background(), client, intent, false, nil)
	if problem != nil || result.Observation != "observed_noop" || service.writes != 0 {
		t.Fatalf("fresh no-op mutated: %+v %v", result, problem)
	}
	service.denied = true
	if _, problem = ApplyRelease(context.Background(), client, intent, true, nil); problem == nil {
		t.Fatal("revoked read authority admitted")
	}
}

func TestUnchangedBaselineRetriesOriginalCASAfterRecordedSend(t *testing.T) {
	service, client := fixture(t)
	intent, problem := PrepareRelease(context.Background(), client, ReleaseRequest{Destination: "alice/model", Release: "v1", Lanes: map[string]string{"fp8": "sha256:" + strings.Repeat("2", 64)}})
	if problem != nil {
		t.Fatal(problem)
	}
	result, problem := ApplyRelease(context.Background(), client, intent, true, func() *exit.Error { return nil })
	if problem != nil || result.Observation != "acknowledged" || service.revision != 2 || service.writes != 1 {
		t.Fatalf("original CAS failed: %+v %v", result, problem)
	}
}
