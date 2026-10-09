package cli

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

// A read the object origin already serves leaves the capability: the machine reads that
// checkpoint anonymously by digest, so the Hub never sees it (th-243). Only a manifest the
// origin does not serve stays a read op, and a served one is asked about once.
func TestReadsOfPublishedCheckpointsLeaveTheCapability(t *testing.T) {
	published, private := strings.Repeat("a", 64), strings.Repeat("b", 64)
	var heads atomic.Int32
	origin := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		heads.Add(1)
		if r.Method == http.MethodHead && r.URL.Path == "/bucket/sha256/"+published {
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer origin.Close()
	read := func(manifest string) map[string]string {
		return map[string]string{"type": "tensorhub_model_read", "model": "alice/m", "manifest": "sha256:" + manifest}
	}
	publish := map[string]string{"type": "tensorhub_model_publish", "model": "alice/out"}
	m := &machineRuns{}
	operations := []any{read(published), read(private), publish}
	kept := m.unpublished(context.Background(), origin.URL+"/bucket", operations)
	if len(kept) != 2 || kept[0].(map[string]string)["manifest"] != "sha256:"+private || kept[1].(map[string]string)["type"] != "tensorhub_model_publish" {
		t.Fatalf("kept %v", kept)
	}
	if len(operations) != 3 {
		t.Fatal("the caller's operations were rewritten")
	}
	before := heads.Load()
	if kept := m.unpublished(context.Background(), origin.URL+"/bucket", []any{read(published)}); len(kept) != 0 || heads.Load() != before {
		t.Fatalf("a published manifest was asked about again: %v", kept)
	}
	if kept := m.unpublished(context.Background(), "", operations); len(kept) != 3 {
		t.Fatal("a Hub with no object origin lost a read")
	}
}
