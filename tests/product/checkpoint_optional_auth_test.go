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
					t.Error("checkpoint read changed credential handling")
				}
				if test.status != 200 {
					w.WriteHeader(test.status)
					_, _ = w.Write([]byte(`{"error":{"code":"proof.refused","message":"refused"}}`))
					return
				}
				if r.Method == http.MethodPost {
					_, _ = w.Write([]byte(`{"reads":[]}`))
				} else {
					_, _ = w.Write([]byte(`{"fixture":"manifest"}`))
				}
			}))
			defer server.Close()
			source := &checkpointTokenSource{token: secret.New(test.machine)}
			client := hub.New(config.Config{HubURL: server.URL, HubToken: secret.New(test.token)}, "checkpoint-auth-proof").WithTokenSource(source)
			ref := hub.Ref{Org: "proof", Name: "source"}
			_, manifestErr := client.CheckpointManifest(context.Background(), ref, "fixture")
			_, readsErr := client.CheckpointReads(context.Background(), ref, "fixture", nil)
			for _, problem := range []*exit.Error{manifestErr, readsErr} {
				if (problem == nil) != (test.status == 200) {
					t.Fatal("checkpoint read did not preserve the Hub result")
				}
				if problem != nil && problem.ErrName() != "proof.refused" {
					t.Fatal("checkpoint refusal was reclassified as login failure")
				}
			}
			if calls.Load() != 2 || source.refreshed.Load() != 0 {
				t.Fatal("optional checkpoint read retried or required credential refresh")
			}
		})
	}
}
