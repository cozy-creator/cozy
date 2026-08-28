package hub

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/config"
	"github.com/cozy-creator/cozy-creator/internal/secret"
)

func TestObserveWorkerSessionUsesRenterBearerAndClosedDeterministicFrameBody(t *testing.T) {
	claim, snapshot, observed := []byte("claim"), []byte("snapshot"), []byte("observed")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/private-rentals/rental-1/worker-session-observations" {
			t.Fatalf("request = %s %s", r.Method, r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer renter-secret" ||
			r.Header.Get("X-Tensorhub-Reason") != "cozy RecordOwner worker session observation" {
			t.Fatalf("relay headers = %#v", r.Header)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil || len(body) != 4 {
			t.Fatalf("body = %#v, %v", body, err)
		}
		for name, want := range map[string][]byte{
			"claim_ack_base64": claim, "snapshot_base64": snapshot, "observed_state_base64": observed,
		} {
			got, err := base64.StdEncoding.DecodeString(body[name].(string))
			if err != nil || string(got) != string(want) {
				t.Fatalf("%s = %q, %v", name, got, err)
			}
		}
		if body["desired_revision"] != float64(7) {
			t.Fatalf("desired_revision = %#v", body["desired_revision"])
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"state": "ready", "desired_revision": 7,
			"accepted_revision": 7, "converged_revision": 7,
		})
	}))
	defer server.Close()

	client := New(config.Config{HubURL: server.URL}, "test").
		WithToken(secret.New("renter-secret"), "test renter token")
	answer, problem := client.ObserveWorkerSession(context.Background(), "rental-1",
		WorkerSessionObservation{ClaimAck: claim, Snapshot: snapshot,
			ObservedState: observed, DesiredRevision: 7})
	if problem != nil || answer.State != RentalReady || answer.ConvergedRevision != 7 {
		t.Fatalf("answer = %#v, %v", answer, problem)
	}
}

func TestObserveWorkerSessionBootFailureIsClosedAlternative(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil || len(body) != 1 ||
			body["boot_failure_base64"] != base64.StdEncoding.EncodeToString([]byte("boot-failure")) {
			t.Fatalf("boot body = %#v, %v", body, err)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"state": "converging"})
	}))
	defer server.Close()
	client := New(config.Config{HubURL: server.URL}, "test").
		WithToken(secret.New("renter-secret"), "test renter token")
	if _, problem := client.ObserveWorkerSession(context.Background(), "rental-1",
		WorkerSessionObservation{BootFailure: []byte("boot-failure")}); problem != nil {
		t.Fatal(problem)
	}
	if _, problem := client.ObserveWorkerSession(context.Background(), "rental-1",
		WorkerSessionObservation{BootFailure: []byte("boot-failure"), DesiredRevision: 1}); problem == nil {
		t.Fatal("mixed boot failure and desired revision was admitted")
	}
}

func TestRentalArtifactGrantUsesScopedBearerAndClosedRequest(t *testing.T) {
	digest := "sha256:0000000000000000000000000000000000000000000000000000000000000001"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/private-rentals/rental-1/artifact-grants" {
			t.Fatalf("request = %s %s", r.Method, r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer renter-secret" {
			t.Fatalf("authorization = %q", got)
		}
		if got := r.Header.Get("X-Tensorhub-Reason"); got != "refresh endpoint closure" {
			t.Fatalf("reason = %q", got)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if len(body) != 2 || body["grant_revision"] != float64(3) || body["ttl_seconds"] != float64(3600) {
			t.Fatalf("body = %#v", body)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"grant_revision": 3,
			"grant": map[string]any{
				"grant_id": "rental-1-grant-3", "expires_at_unix": time.Now().Add(time.Hour).Unix(),
				"subjects": []map[string]any{{
					"digest": digest, "subject_id": digest, "kind": "plan", "length": 17,
				}},
				"locations": []map[string]any{{"digest": digest, "url": "https://objects.invalid/exact"}},
			},
		})
	}))
	defer server.Close()

	client := New(config.Config{HubURL: server.URL, HubToken: secret.New("admin")}, "test").
		WithToken(secret.New("renter-secret"), "test renter token")
	grant, problem := client.ArtifactGrant(context.Background(), "rental-1", 3, 3600,
		"refresh endpoint closure")
	if problem != nil {
		t.Fatal(problem)
	}
	if grant.GrantRevision != 3 || grant.Grant.GrantID != "rental-1-grant-3" ||
		len(grant.Grant.Subjects) != 1 || len(grant.Grant.Locations) != 1 {
		t.Fatalf("grant = %#v", grant)
	}
}

func TestReviseRentalPlacementUsesAdminMutationContract(t *testing.T) {
	control := []byte(`{"format":"tensorhub.rental_control_snapshot/1"}`)
	placement := []byte(`{"format":"cozy.worker.v1.PlacementSet/2"}`)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/admin/private-rentals/rental-1/placement-revisions" {
			t.Fatalf("request = %s %s", r.Method, r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer admin" ||
			r.Header.Get("Idempotency-Key") != "revise-1" ||
			r.Header.Get("X-Tensorhub-Reason") != "switch endpoint" {
			t.Fatalf("mutation headers = %#v", r.Header)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil || len(body) != 1 ||
			body["endpoint_ref"] != "cozy/endpoint/v2/generate" {
			t.Fatalf("body = %#v, %v", body, err)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"rental_id": "rental-1", "endpoint_ref": "cozy/endpoint/v2/generate",
			"placement_revision": 2,
			"control_snapshot":   map[string]any{"canonical_bytes": control, "digest": digestOf(control), "length": len(control)},
			"placement_set":      map[string]any{"canonical_bytes": placement, "digest": digestOf(placement), "length": len(placement)},
		})
	}))
	defer server.Close()
	client := New(config.Config{HubURL: server.URL, HubToken: secret.New("admin")}, "test")
	revision, problem := client.ReviseRentalPlacement(context.Background(), "rental-1",
		"cozy/endpoint/v2/generate", "revise-1", "switch endpoint")
	if problem != nil || revision.PlacementRevision != 2 ||
		string(revision.PlacementSet.CanonicalBytes) != string(placement) {
		t.Fatalf("revision = %#v, %v", revision, problem)
	}
}
