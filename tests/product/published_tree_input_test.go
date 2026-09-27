package producttest

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/resultfiles"
	"github.com/cozy-creator/cozy/internal/secret"
)

type publishedTreeResolver struct {
	publishedRouteResolver
	entry *launch.Entrypoint
	calls int
}

func (r *publishedTreeResolver) ResolveRemoteJob(hub, pkg, release, function string, models []orchestrator.ModelRef, deferred bool) (orchestrator.LogicalJob, *launch.Entrypoint, *exit.Error) {
	r.calls++
	return orchestrator.LogicalJob{Package: pkg, Release: release, Function: function, DescriptorID: childDigest("9")}, r.entry, nil
}

func TestPublishedPrivateTreeUsesResolvedSchemaAndImmutableSnapshot(t *testing.T) {
	o := hostOwner(t, "published-tree-input")
	fatal(t, o.store.RecordRental(records.Rental{ID: "rental-tree", MachineName: "tree-host", SKU: "cpu", AcceleratorModel: "CPU", State: "ready", Hub: "http://127.0.0.1:1", Address: "127.0.0.1:1", AcceleratorCount: 1, HourlyRateUSDMicros: 1}))
	resolver := &publishedTreeResolver{entry: treeArgumentEntry(t)}
	const bearer = "published-tree-input-fixture"
	handler, problem := api.New(api.Options{Orchestrator: o.c, Cfg: o.cfg, Creds: api.Credentials{CLI: secret.New(bearer)}, Addr: "127.0.0.1:11111", Web: http.NotFoundHandler(), Packages: resolver, MachineExecutions: publishedRouteObserver{}}).Handler()
	fatal(t, problem)
	source := t.TempDir()
	original := []byte("retained prefix")
	must(t, os.WriteFile(filepath.Join(source, "prefix.txt"), original, 0600))
	submitted, err := json.Marshal(map[string]any{"package": "alice/ops", "release": "1.0.0", "function": "main", "input": map[string]any{"resume_from": "prefix"}, "rental": true, "requested_rental": "rental-tree", "trees": []string{"prefix=" + source}})
	must(t, err)
	request := httptest.NewRequest(http.MethodPost, "http://127.0.0.1:11111/v1/local/jobs", bytes.NewReader(submitted))
	request.RemoteAddr = "127.0.0.1:12345"
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Authorization", "Bearer "+bearer)
	request.Header.Set("Idempotency-Key", "prefix")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusAccepted {
		t.Fatalf("published Tree input %d: %s", response.Code, response.Body.String())
	}
	row, problem := o.store.RequestByIdempotencyKey("prefix")
	fatal(t, problem)
	if row == nil || len(row.Assets) != 1 || row.Assets[0].Snapshot == nil || row.LocalInstallationID != "" || len(row.Trees) != 0 || resolver.calls != 1 {
		t.Fatalf("published Tree did not use exactly one resolved native intake: row=%+v metadata_reads=%d", row, resolver.calls)
	}
	link, problem := o.store.MachineExecution(row.ID)
	fatal(t, problem)
	if link == nil {
		t.Fatal("published Tree fell back to client coordination")
	}
	must(t, os.WriteFile(filepath.Join(source, "prefix.txt"), []byte("edited"), 0600))
	input := row.Assets[0]
	members, problem := resultfiles.ReadTreeManifest(input.Snapshot.Path, input.Digest, input.Length, input.Snapshot.ContentBytes)
	fatal(t, problem)
	if len(members) != 1 || members[0].Path != "prefix.txt" {
		t.Fatal("native snapshot lost its source closure")
	}
	content, err := os.ReadFile(filepath.Join(input.Snapshot.Path+".files", strings.TrimPrefix(members[0].Digest, "sha256:")))
	must(t, err)
	if !bytes.Equal(content, original) {
		t.Fatal("published input followed a mutated directory")
	}
}
