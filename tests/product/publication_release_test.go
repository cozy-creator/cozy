package producttest

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/publication"
	"github.com/cozy-creator/cozy/internal/secret"
)

// The owner client speaks its real HTTP API. This fixture controls commits and competing
// revisions, without replacing publication logic with a fake.
type publicationReleaseServer struct {
	mu       sync.Mutex
	revision int64
	lanes    map[string]string
	writes   int
	denied   bool
}

func (s *publicationReleaseServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
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
	lanes := []hub.ModelReleaseLane{}
	for name, id := range s.lanes {
		lanes = append(lanes, hub.ModelReleaseLane{Lane: name, CheckpointID: id})
	}
	json.NewEncoder(w).Encode(hub.ModelRelease{Release: "v1", Revision: s.revision, Lanes: lanes, Changed: true})
}

func publicationReleaseFixture(t *testing.T) (*publicationReleaseServer, *hub.Client) {
	t.Helper()
	service := &publicationReleaseServer{revision: 1, lanes: map[string]string{"bf16": "sha256:" + strings.Repeat("1", 64)}}
	server := httptest.NewServer(service)
	t.Cleanup(server.Close)
	return service, hub.New(config.Config{HubURL: server.URL, HubToken: secret.New("test")}, "publication-test")
}

func TestExplicitStaleRevisionAndFreshNoop(t *testing.T) {
	service, client := publicationReleaseFixture(t)
	desired := map[string]string{"bf16": service.lanes["bf16"]}
	stale := int64(0)
	_, problem := publication.PrepareRelease(context.Background(), client, publication.ReleaseRequest{Destination: "alice/model", Release: "v1", Lanes: desired, ExpectedRevision: &stale})
	if problem == nil || problem.ErrName() != "publication.release_conflict" {
		t.Fatal("explicit stale no-op admitted", problem)
	}
	intent, problem := publication.PrepareRelease(context.Background(), client, publication.ReleaseRequest{Destination: "alice/model", Release: "v1", Lanes: desired})
	if problem != nil {
		t.Fatal(problem)
	}
	result, problem := publication.ApplyRelease(context.Background(), client, intent, false, nil)
	if problem != nil || result.Observation != "observed_noop" || service.writes != 0 {
		t.Fatalf("fresh no-op mutated: %+v %v", result, problem)
	}
	service.denied = true
	if _, problem = publication.ApplyRelease(context.Background(), client, intent, true, nil); problem == nil {
		t.Fatal("revoked read authority admitted")
	}
}

func TestUnchangedBaselineRetriesOriginalCASAfterRecordedSend(t *testing.T) {
	service, client := publicationReleaseFixture(t)
	intent, problem := publication.PrepareRelease(context.Background(), client, publication.ReleaseRequest{Destination: "alice/model", Release: "v1", Lanes: map[string]string{"fp8": "sha256:" + strings.Repeat("2", 64)}})
	if problem != nil {
		t.Fatal(problem)
	}
	result, problem := publication.ApplyRelease(context.Background(), client, intent, true, func() *exit.Error { return nil })
	if problem != nil || result.Observation != "acknowledged" || service.revision != 2 || service.writes != 1 {
		t.Fatalf("original CAS failed: %+v %v", result, problem)
	}
}
