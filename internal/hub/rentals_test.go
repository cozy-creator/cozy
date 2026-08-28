package hub

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/config"
	"github.com/cozy-creator/cozy-creator/internal/secret"
)

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
