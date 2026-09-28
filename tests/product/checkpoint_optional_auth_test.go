package producttest

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/secret"
)

type checkpointTokenSource struct {
	token     secret.Value
	refreshed atomic.Int32
}

func (s *checkpointTokenSource) AccessToken(context.Context) (secret.Value, *exit.Error) {
	if s.token.Present() {
		return s.token, nil
	}
	return secret.Value{}, exit.Named(exit.Credential, "auth.machine_key_missing", "not enrolled")
}
func (s *checkpointTokenSource) Invalidate() { s.refreshed.Add(1) }

// Unpublished checkpoints are private to their account (Tensorhub #852): every by-digest
// read, including model resolution and rental quotes, presents the owner's bearer when one
// exists and reads anonymously otherwise.
func TestCheckpointReadsUseOptionalOwnerCredentials(t *testing.T) {
	for _, test := range []struct {
		name, token, machine, authorization string
		status                              int
	}{
		{"signed out public", "", "", "", 200},
		{"owner fallback", "owner", "", "Bearer owner", 200},
		{"machine credential", "", "owner", "Bearer owner", 200},
		{"machine credential preferred", "operator", "owner", "Bearer owner", 200},
		{"anonymous missing", "", "", "", 404},
		{"anonymous unauthorized", "", "", "", 401},
		{"owner denied", "wrong", "", "Bearer wrong", 403},
		{"server failure", "owner", "", "Bearer owner", 503},
	} {
		t.Run(test.name, func(t *testing.T) {
			var calls atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				if r.Header.Get("Authorization") != test.authorization {
					t.Errorf("%s changed credential handling: %q", r.URL.Path, r.Header.Get("Authorization"))
				}
				if test.status != 200 {
					w.WriteHeader(test.status)
					_, _ = w.Write([]byte(`{"error":{"code":"proof.refused","message":"refused"}}`))
					return
				}
				switch {
				case r.URL.Path == "/v1/models/resolve":
					_, _ = w.Write([]byte(`{"model":"proof/source","manifest_id":"sha256:fixture"}`))
				case r.URL.Path == "/v1/rental-quotes":
					_, _ = w.Write([]byte(`{"price_usd_micros_per_hour":1}`))
				case r.Method == http.MethodPost:
					_, _ = w.Write([]byte(`{"reads":[]}`))
				default:
					_, _ = w.Write([]byte(`{"fixture":"manifest"}`))
				}
			}))
			defer server.Close()
			source := &checkpointTokenSource{token: secret.New(test.machine)}
			client := hub.New(config.Config{HubURL: server.URL, HubToken: secret.New(test.token)}, "checkpoint-auth-proof").WithTokenSource(source)
			ref := hub.Ref{Org: "proof", Name: "source"}
			_, manifestErr := client.CheckpointManifest(context.Background(), ref, "fixture")
			_, readsErr := client.CheckpointReads(context.Background(), ref, "fixture", nil)
			_, resolveErr := client.ResolveModel(context.Background(), "proof/source#sha256:fixture", "")
			_, quoteErr := client.QuoteRental(context.Background(), []byte(`{}`))
			for _, problem := range []*exit.Error{manifestErr, readsErr, resolveErr, quoteErr} {
				if (problem == nil) != (test.status == 200) {
					t.Fatal("checkpoint read did not preserve the Hub result")
				}
				if problem != nil && problem.ErrName() != "proof.refused" {
					t.Fatal("checkpoint refusal was reclassified as login failure")
				}
			}
			if calls.Load() != 4 || source.refreshed.Load() != 0 {
				t.Fatal("optional checkpoint read retried or required credential refresh")
			}
		})
	}
}
