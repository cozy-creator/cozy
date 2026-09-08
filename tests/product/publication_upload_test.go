package producttest

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/publication"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/secret"
)

func TestCheckpointLostFinalizeReplyAndFreshEffectReuseRetainedManifest(t *testing.T) {
	manifest := []byte(`{"entries":[]}`)
	digest, _ := canonical.Spell(canonical.Digest(manifest))
	object := hub.Object{ID: digest, Length: int64(len(manifest))}
	intent := publication.UploadIntent{Request: publication.UploadRequest{Destination: "alice/model", Artifact: records.ModelArtifact{ProducerRequestID: "source", OutputSlot: "model", Manifest: records.ArtifactObjectRef{Digest: digest, Length: int64(len(manifest))}, TensorFSReceiptDigest: "sha256:" + strings.Repeat("1", 64)}}, Objects: []hub.Object{object}}
	accepted, checkpointed := false, false
	puts, finalizes, opens := 0, 0, 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodGet:
			if !checkpointed {
				http.Error(w, `{"error":{"code":"not_found","message":"absent"}}`, 404)
				return
			}
			w.Write(manifest)
		case r.Method == http.MethodPut:
			opens++
			state := "claimed"
			if accepted {
				state = "accepted"
			}
			json.NewEncoder(w).Encode(hub.OpenPublicationResponse{Publication: hub.Session{Operation: "effect-one", State: "open", Objects: []hub.Transfer{{ObjectID: digest, Length: object.Length, State: state}}}})
		case strings.HasSuffix(r.URL.Path, "/grants"):
			json.NewEncoder(w).Encode(hub.GrantResponse{ServerTimeUnix: 100, Grants: []hub.Grant{{ObjectID: digest, Length: object.Length, URL: "http://object.invalid/object", ExpiresAtUnix: 200}}})
		case strings.HasSuffix(r.URL.Path, "/finalize"):
			finalizes++
			if !accepted {
				t.Error("finalization preceded object acceptance")
			}
			checkpointed = true
			connection, _, err := w.(http.Hijacker).Hijack()
			if err != nil {
				t.Error(err)
				return
			}
			connection.Close()
		default:
			t.Errorf("unexpected publication path %s", r.URL.Path)
		}
	}))
	defer server.Close()
	client := hub.New(config.Config{HubURL: server.URL, HubToken: secret.New("test")}, "publication-test")
	upload := func(context.Context, hub.Grant, int64) *exit.Error { puts++; accepted = true; return nil }
	before := func() *exit.Error { return nil }
	if _, problem := publication.UploadCheckpoint(context.Background(), client, "effect-one", intent, before, upload); problem == nil {
		t.Fatal("lost finalize response was falsely acknowledged")
	}
	receipt, problem := publication.UploadCheckpoint(context.Background(), client, "effect-one", intent, before, upload)
	if problem != nil || receipt.Observation != "observed_convergence" {
		t.Fatalf("lost finalize did not reconcile: %+v %v", receipt, problem)
	}
	fresh, problem := publication.UploadCheckpoint(context.Background(), client, "effect-two", intent, before, upload)
	if problem != nil || fresh.Checkpoint != digest || fresh.Observation != "observed_convergence" {
		t.Fatalf("fresh same-content effect failed: %+v %v", fresh, problem)
	}
	if puts != 1 || finalizes != 1 || opens != 1 {
		t.Fatalf("replayed checkpoint repeated transfer: puts%d finalizes%d opens%d", puts, finalizes, opens)
	}
}
