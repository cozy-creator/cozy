package producttest

import (
	"bytes"
	"encoding/json"
	"flag"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

var machinePublicationScript = flag.String("machine-publication-script", "", "captured script for the actual scoped Host publication proof")
var machinePublicationIdentity = flag.String("machine-publication-identity", "", "protected actual Host identity handoff path")
var machinePublicationHandoffPath = flag.String("machine-publication-handoff", "", "protected real Hub handoff path")

const machinePublicationCanary = "cl259-private-script-source-must-never-be-published"

type machinePublicationHandoff struct {
	APIOrigin    string `json:"api_origin"`
	WorkerOrigin string `json:"worker_origin"`
	WorkerToken  string `json:"worker_token"`
	Access       string `json:"creator_access_token"`
	RentalID     string `json:"rental_id"`
	WorkerID     string `json:"worker_id"`
	Destination  string `json:"destination"`
}

func bindMachinePublicationFixture(t *testing.T, layout home.Layout, host actualChildHost, catalog *fakeRentalHub, creatorPublic, mediaHash string) string {
	t.Helper()
	identityPath := *machinePublicationIdentity
	handoffPath := *machinePublicationHandoffPath
	if identityPath == "" && handoffPath == "" {
		return "rental-idle-test"
	}
	if identityPath == "" || handoffPath == "" {
		t.Fatal("publication fixture needs both protected identity and handoff paths")
	}
	key := filepath.Join(layout.Root, "host-worker.key")
	if out, err := exec.Command("docker", "cp", host.Container+":/run/cozy/bootstrap/tls.key", key).CombinedOutput(); err != nil {
		t.Fatalf("read owned Host identity: %v %s", err, out)
	}
	must(t, os.Chmod(key, 0600))
	identity, err := json.Marshal(map[string]any{"rental_id": "rental-private-child-host", "worker_id": "private-child-host", "creator_public_key": creatorPublic, "media_token_sha256": mediaHash, "certificate": filepath.Join(layout.Root, "host-certificate.pem"), "key": key})
	must(t, err)
	must(t, os.MkdirAll(filepath.Dir(identityPath), 0700))
	must(t, os.WriteFile(identityPath, identity, 0600))
	var handoff machinePublicationHandoff
	for deadline := time.Now().Add(3 * time.Minute); ; {
		raw, err := os.ReadFile(handoffPath)
		if err == nil && json.Unmarshal(raw, &handoff) == nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("real Hub did not supply its protected publication handoff")
		}
		time.Sleep(100 * time.Millisecond)
	}
	if handoff.RentalID != "rental-private-child-host" || handoff.WorkerID != "private-child-host" || handoff.Access == "" || len(handoff.WorkerToken) != 43 || handoff.WorkerOrigin == "" || handoff.Destination != "machineproof/model" {
		t.Fatal("real Hub handoff differs from the actual Host identity or requested scope")
	}
	authority, err := exec.Command("docker", "exec", host.Container, "cat", "/run/cozy/bootstrap/machine-publication-authority.json").Output()
	must(t, err)
	var actual struct {
		Version     int    `json:"version"`
		Origin      string `json:"origin"`
		WorkerID    string `json:"worker_id"`
		WorkerToken string `json:"worker_token"`
	}
	must(t, json.Unmarshal(authority, &actual))
	if actual.Version != 2 || actual.Origin != handoff.WorkerOrigin || actual.WorkerID != handoff.WorkerID || actual.WorkerToken != handoff.WorkerToken {
		t.Fatal("Host boot authority differs from its provisioned real Hub origin/worker capability")
	}
	origin, err := url.Parse(handoff.APIOrigin)
	must(t, err)
	if origin.Scheme != "http" || origin.Hostname() != "127.0.0.1" {
		t.Fatal("the real Hub fixture must bind its public API to loopback")
	}
	proxy := &httputil.ReverseProxy{Rewrite: func(request *httputil.ProxyRequest) { request.SetURL(origin) }}
	provider := catalog.server.Config.Handler
	catalog.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		if request.URL.Path == "/v1/machine-authorizations" {
			proxy.ServeHTTP(w, request) // Real AuthKit authorization, never a test response.
			return
		}
		if request.Header.Get("Authorization") != "Bearer "+handoff.Access {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		copy := request.Clone(request.Context())
		copy.Header.Set("Authorization", "Bearer rental-idle-test")
		provider.ServeHTTP(w, copy) // Only provider inventory/readback is a fixture.
	})
	return handoff.Access
}

func TestMachinePublicationActualHostAfterClientDisconnect(t *testing.T) {
	handoff := *machinePublicationHandoffPath
	if *machinePublicationScript == "" || handoff == "" {
		t.Skip("requires an explicit captured publication script and real Hub handoff")
	}
	completed := false
	t.Cleanup(func() {
		if !completed {
			_ = os.WriteFile(handoff+".done", []byte(`{"error":"Creator CLI publication proof failed; inspect its private test log"}`), 0600)
		}
	})
	layout, store, host, path, _ := startActualChildHost(t)
	script, err := os.ReadFile(*machinePublicationScript)
	must(t, err)
	if !bytes.Contains(script, []byte(machinePublicationCanary)) {
		t.Fatal("publication proof script has no private source canary")
	}
	code, out := runCozyPath(t, layout.Root, path, "run", *machinePublicationScript, "--rental", "child-host", "--allow-publish", "machineproof/model", "--json", "--idempotency-key", "machine-publication-offline")
	if code != 0 || !strings.Contains(out, `"machine_accepted":true`) {
		t.Fatalf("scoped root acceptance [%d]: %s", code, out)
	}
	request, problem := store.RequestByIdempotencyKey("machine-publication-offline")
	fatal(t, problem)
	intentBytes, problem := store.MachinePublicationIntent(request.ID)
	fatal(t, problem)
	var intent hub.MachinePublicationGrantIntent
	must(t, json.Unmarshal(intentBytes, &intent))
	link, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	var receipt pb.MachineExecutionReceipt
	must(t, proto.Unmarshal(link.Receipt, &receipt))
	if receipt.PublicationAuthorizationId == "" || receipt.PublicationAuthorizationId != intent.AuthorizationID {
		t.Fatal("actual Runtime accepted a different publication authority")
	}
	code, out = runCozyPath(t, layout.Root, path, "down", "--json")
	if code != 0 || daemonOnRoot(layout.Root) != 0 {
		t.Fatalf("client disconnect before publication [%d]: %s", code, out)
	}
	for deadline := time.Now().Add(4 * time.Minute); ; {
		raw, err := exec.Command("docker", "exec", host.Container, "python3", "-c", `import json,sqlite3,sys
from cozy_runtime.internal.placement_materialization import LOCAL_TENSORFS_ROOT
c=sqlite3.connect('file:'+str(LOCAL_TENSORFS_ROOT/'.cozy-workspace/journal.sqlite3')+'?mode=ro',uri=True)
print(json.dumps(c.execute('select state,collected from executions where request=?',(sys.argv[1],)).fetchone()))`, request.ID).Output()
		must(t, err)
		if daemonOnRoot(layout.Root) != 0 {
			t.Fatal("client daemon returned before independent publication finished")
		}
		var state []any
		must(t, json.Unmarshal(raw, &state))
		if len(state) == 2 && state[0] == "succeeded" && state[1] == float64(0) {
			must(t, os.WriteFile(filepath.Join(layout.Root, "publication-offline.json"), raw, 0600))
			break
		}
		if time.Now().After(deadline) || len(state) != 2 || state[0] == "failed" || state[0] == "canceled" {
			t.Fatalf("independent publication did not complete: %s", raw)
		}
		time.Sleep(250 * time.Millisecond)
	}
	code, out = runCozyPath(t, layout.Root, path, "run", "watch", request.ID, "--json")
	if code != 0 {
		t.Fatalf("collect published result [%d]: %s", code, out)
	}
	var response struct{ Result struct{ Value string } }
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		_ = json.Unmarshal([]byte(line), &response)
	}
	var value struct {
		Checkpoint, Release, Lane string
		Revision                  int
	}
	must(t, json.Unmarshal([]byte(response.Result.Value), &value))
	if value.Checkpoint == "" || value.Release != "v1" || value.Revision != 1 || value.Lane != "fp16" {
		t.Fatal("publication returned no exact checkpoint/release receipt")
	}
	attempts, problem := store.Attempts(request.ID)
	fatal(t, problem)
	if len(attempts) != 0 {
		t.Fatal("Creator coordinated publication execution attempts")
	}
	processes, err := exec.Command("docker", "exec", host.Container, "python3", "-c", `import json
from pathlib import Path
found=[]
for path in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  argv=path.read_bytes().decode().split('\0')
  if any(Path(item).name in ('cozy','cozy-daemon') for item in argv[:2]): found.append(argv[:2])
 except (OSError,ProcessLookupError): pass
print(json.dumps(found))`).Output()
	must(t, err)
	if strings.TrimSpace(string(processes)) != "[]" {
		t.Fatal("a Creator process ran inside the publication Host")
	}
	done, err := json.Marshal(map[string]any{"grant_id": intent.AuthorizationID, "checkpoint": value.Checkpoint, "release": value.Release, "lane": value.Lane, "revision": value.Revision, "private_source_canary": machinePublicationCanary})
	must(t, err)
	must(t, os.WriteFile(handoff+".done", done, 0600))
	completed = true
	t.Logf("scoped Runtime publication completed while client absent: request=%s checkpoint=%s release=%s revision=%d", request.ID, value.Checkpoint, value.Release, value.Revision)
}
