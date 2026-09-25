package producttest

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

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

func (r *publishedTreeResolver) ResolveRemoteJob(pkg, release, function string, models []orchestrator.ModelRef, deferred bool) (orchestrator.LogicalJob, *launch.Entrypoint, *exit.Error) {
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
	if row == nil || len(row.Assets) != 1 || row.Assets[0].Snapshot == nil || row.LocalPackageDigest != "" || len(row.Trees) != 0 || resolver.calls != 1 {
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

func TestPublishedOptionalTreeDirectoryActualHost(t *testing.T) {
	proof := startPublishedMachineHost(t)
	layout, store, host, path := proof.Layout, proof.Store, proof.Host, proof.Path
	source := t.TempDir()
	original := []byte("retained published prefix")
	must(t, os.WriteFile(filepath.Join(source, "prefix.txt"), original, 0600))
	readExport := func(request string, expected []byte) {
		t.Helper()
		exported, problem := store.OutputExportOf(request)
		fatal(t, problem)
		if exported == nil || exported.State != "published" || len(exported.PublishedPaths) != 1 {
			t.Fatal("published prefix was not exported")
		}
		data, err := os.ReadFile(filepath.Join(exported.PublishedPaths[0], "prefix.txt"))
		must(t, err)
		if !bytes.Equal(data, expected) {
			t.Fatalf("prefix bytes changed: %q", data)
		}
	}
	for _, present := range []bool{false, true} {
		key := "without-prefix"
		args := []string{"run", proof.Package.Package + "/main", "--rental", "child-host", "--json"}
		if present {
			key = "with-prefix"
			args = append(args, "--asset", "prefix="+source)
		} else {
			args = append(args, "--await")
		}
		args = append(args, "--idempotency-key", key)
		code, output := runCozyPath(t, layout.Root, path, args...)
		if code != 0 {
			t.Fatalf("published optional Tree intake [%d]: %s", code, output)
		}
		request, problem := store.RequestByIdempotencyKey(key)
		fatal(t, problem)
		if request == nil || request.LocalPackageDigest != "" {
			t.Fatal("Tree input changed published code identity")
		}
		if !present {
			readExport(request.ID, []byte("empty prefix"))
			continue
		}
		if len(request.Assets) != 1 || request.Assets[0].Snapshot == nil {
			t.Fatal("published Tree has no native input snapshot")
		}
		must(t, os.WriteFile(filepath.Join(source, "prefix.txt"), []byte("edited original"), 0600))
		if code, output := runCozyPath(t, layout.Root, path, "down", "--json"); code != 0 || daemonOnRoot(layout.Root) != 0 {
			t.Fatalf("detach published Tree client [%d]: %s", code, output)
		}
		probe := func() []byte {
			t.Helper()
			data, err := exec.Command("docker", "--context", "default", "exec", host.Container, "python3", "-c", `import json,sqlite3,sys
from cozy_runtime.internal.placement_materialization import LOCAL_TENSORFS_ROOT
c=sqlite3.connect('file:'+str(LOCAL_TENSORFS_ROOT/'.cozy-workspace/journal.sqlite3')+'?mode=ro',uri=True,timeout=10)
c.row_factory=sqlite3.Row
r=sys.argv[1]
v=dict(c.execute('select request,state,collected from executions where request=?',(r,)).fetchone())
v['released_inputs']=c.execute("select count(*) from input_tree_intakes where request=? and state='released'",(r,)).fetchone()[0]
print(json.dumps(v))`, request.ID).CombinedOutput()
			if err != nil {
				t.Fatalf("read published native input: %v %s", err, data)
			}
			return data
		}
		for deadline := time.Now().Add(3 * time.Minute); ; {
			raw := probe()
			var state struct {
				State     string
				Collected int
			}
			must(t, json.Unmarshal(raw, &state))
			if state.State == "succeeded" && state.Collected == 0 {
				break
			}
			if state.State == "failed" || state.State == "canceled" || time.Now().After(deadline) {
				t.Fatalf("private published job did not finish with client absent: %s", raw)
			}
			time.Sleep(200 * time.Millisecond)
		}
		if code, output := runCozyPath(t, layout.Root, path, "run", "watch", request.ID, "--json"); code != 0 {
			t.Fatalf("collect published prefix [%d]: %s", code, output)
		}
		readExport(request.ID, original)
		attempts, problem := store.Attempts(request.ID)
		fatal(t, problem)
		children, problem := store.Children(request.ID)
		fatal(t, problem)
		if len(attempts) != 0 || len(children) != 0 {
			t.Fatal("Creator owns published Tree execution")
		}
		final := probe()
		var state struct {
			State     string
			Collected int
			Released  int `json:"released_inputs"`
		}
		must(t, json.Unmarshal(final, &state))
		if state.State != "succeeded" || state.Collected != 1 || state.Released != 1 {
			t.Fatalf("published Tree native input not released: %s", final)
		}
		must(t, os.WriteFile(filepath.Join(layout.Root, "published-tree-proof.json"), final, 0600))
		t.Logf("published optional prefix absent and present; original edited, client detached; %s", final)
	}
}
