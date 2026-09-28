package producttest

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/secret"
)

func TestCheckpointDeleteAcceptsExactAlreadyAbsentConfirmation(t *testing.T) {
	checkpoint := "sha256:" + strings.Repeat("1", 64)
	repository := "sha256:" + strings.Repeat("2", 64)
	for _, test := range []struct {
		name, checkpoint, repository string
		removed, wantError           bool
	}{
		{name: "removed", checkpoint: checkpoint, repository: repository, removed: true},
		{name: "already absent", checkpoint: checkpoint, repository: repository},
		{name: "wrong checkpoint", checkpoint: repository, repository: repository, wantError: true},
		{name: "missing canonical repo", checkpoint: checkpoint, wantError: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodDelete || r.Header.Get("Authorization") != "Bearer proof" ||
					r.URL.Path != "/v1/models/proof/tiny/checkpoints/"+checkpoint {
					t.Error("wrong authenticated checkpoint removal request")
				}
				_ = json.NewEncoder(w).Encode(map[string]any{"checkpoint_id": test.checkpoint,
					"repository_sha256": test.repository, "removed": test.removed})
			}))
			defer server.Close()
			client := hub.New(config.Config{HubURL: server.URL}, "checkpoint-delete-proof").WithToken(secret.New("proof"), "test")
			ref, problem := hub.ParseRef("proof/tiny")
			fatal(t, problem)
			problem = client.RemoveCheckpoint(t.Context(), ref, checkpoint, "remove exact fixture")
			if (problem != nil) != test.wantError {
				t.Fatalf("removal result=%v, want error=%v", problem, test.wantError)
			}
		})
	}
}
