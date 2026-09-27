package producttest

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/records"
)

var machineExecutionScript = flag.String("machine-execution-script", "", "captured script with an exact coherent Runtime dependency for the actual Host proof")
var machineExecutionOfflineScript = flag.String("machine-execution-offline-script", "", "captured slow script for ordinary CLI disconnection from the actual Host")

func TestMachineExecutionActualHostDisconnect(t *testing.T) {
	if *machineExecutionOfflineScript == "" {
		t.Skip("requires an explicit slow script and actual Host cohort")
	}
	layout, store, host, path, _ := startActualChildHost(t)
	key := fmt.Sprintf("machine-offline-%d", time.Now().UnixNano())
	code, out := runCozyPath(t, layout.Root, path, "run", *machineExecutionOfflineScript, "--rental", "child-host", "--json", "--idempotency-key", key)
	if code != 0 || !strings.Contains(out, `"machine_accepted":true`) {
		t.Fatalf("private execution was not durably accepted [%d]: %s", code, out)
	}
	request, problem := store.RequestByIdempotencyKey(key)
	fatal(t, problem)
	if request == nil || records.Settled(request.State) {
		t.Fatal("slow execution settled before the disconnect proof")
	}
	code, out = runCozyPath(t, layout.Root, path, "down", "--json")
	if code != 0 || daemonOnRoot(layout.Root) != 0 {
		t.Fatalf("ordinary client disconnect [%d]: %s", code, out)
	}
	var offline []byte
	for deadline := time.Now().Add(2 * time.Minute); ; {
		// Read the actual worker journal while the client is absent. This is
		// evidence only; submission/control/collection all use the ordinary CLI.
		var err error
		offline, err = exec.Command("docker", "exec", host.Container, "python3", "-c", `import json,sqlite3,sys
from cozy_runtime.internal.placement_materialization import LOCAL_TENSORFS_ROOT
c=sqlite3.connect('file:'+str(LOCAL_TENSORFS_ROOT/'.cozy-workspace/journal.sqlite3')+'?mode=ro',uri=True)
c.row_factory=sqlite3.Row
print(json.dumps(dict(c.execute('select request,state,collected,ordinal,generation,accepted_ms,finished_ms from executions where request=?',(sys.argv[1],)).fetchone())))`, request.ID).Output()
		must(t, err)
		var state struct {
			State     string `json:"state"`
			Collected int    `json:"collected"`
		}
		must(t, json.Unmarshal(offline, &state))
		if daemonOnRoot(layout.Root) != 0 {
			t.Fatal("client daemon returned before disconnected completion")
		}
		if state.State == "succeeded" && state.Collected == 0 {
			break
		}
		if time.Now().After(deadline) || state.State == "failed" || state.State == "canceled" {
			t.Fatalf("private execution did not finish with its client absent: %s", offline)
		}
		time.Sleep(250 * time.Millisecond)
	}
	must(t, os.WriteFile(filepath.Join(layout.Root, request.ID+"-offline.json"), offline, 0600))
	code, out = runCozyPath(t, layout.Root, path, "run", "watch", request.ID, "--json")
	if code != 0 {
		t.Fatalf("reconnect and collect private execution [%d]: %s", code, out)
	}
	link, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	attempts, problem := store.Attempts(request.ID)
	fatal(t, problem)
	if link == nil || !link.Collected || len(attempts) != 0 {
		t.Fatal("private reconnect did not collect the Runtime result without client attempts")
	}
	t.Logf("ordinary CLI private Host disconnected completion and collection: %s", offline)
}

func TestMachineExecutionActualHostRoot(t *testing.T) {
	if *machineExecutionScript == "" {
		t.Skip("requires an explicit captured script and actual Host cohort")
	}
	layout, store, host, path, _ := startActualChildHost(t)
	key := fmt.Sprintf("machine-host-root-%d", time.Now().UnixNano())
	status, out := runCozyPath(t, layout.Root, path, "run", *machineExecutionScript,
		"--rental", "child-host", "--await", "--json", "--idempotency-key", key)
	if status != 0 {
		t.Fatalf("Runtime-owned private Host invocation [%d]: %s", status, out)
	}
	request, problem := store.RequestByIdempotencyKey(key)
	fatal(t, problem)
	if request == nil || request.State != "succeeded" {
		t.Fatalf("private root did not succeed: %+v", request)
	}
	link, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	if link == nil || link.MachineID != "rental-private-child-host" || len(link.Receipt) == 0 || !link.Collected {
		t.Fatalf("private root lacks its exact Runtime receipt and collection: %+v", link)
	}
	attempts, problem := store.Attempts(request.ID)
	fatal(t, problem)
	children, problem := store.Children(request.ID)
	fatal(t, problem)
	if len(attempts) != 0 || len(children) != 0 {
		t.Fatalf("client owns machine attempts or children: attempts=%d children=%d", len(attempts), len(children))
	}
	// Inspect actual processes inside this owned container. The Host and Runtime
	// implement execution; no Creator daemon may be running alongside them.
	processes, err := exec.Command("docker", "exec", host.Container, "python3", "-c", `import json
from pathlib import Path
found=[]
for path in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  argv=path.read_bytes().decode().split('\0')
  found.append(argv[:2])
 except (OSError,ProcessLookupError): pass
print(json.dumps(found))
`).CombinedOutput()
	must(t, err)
	var commands [][]string
	must(t, json.Unmarshal(processes, &commands))
	for _, command := range commands {
		for _, argument := range command {
			if argument == "cozy" || argument == "cozy-daemon" || strings.HasSuffix(argument, "/cozy") || strings.HasSuffix(argument, "/cozy-daemon") {
				t.Fatalf("Creator process is running inside private Host: %s", processes)
			}
		}
	}
	t.Logf("private Runtime root %s accepted and collected; no Creator attempts or children; %s", request.ID, out)
	status, out = runCozyPath(t, layout.Root, path, "run", *machineExecutionScript,
		"--rental", "child-host", "--await", "--json", "--idempotency-key", key)
	if status != 0 {
		t.Fatalf("identical completed capture replay [%d]: %s", status, out)
	}

	status, out = runCozyPath(t, layout.Root, path, "run", *machineExecutionScript,
		"--rental", "child-host", "--allow-upload", "alice/model", "--json", "--idempotency-key", key+"-publication-refused")
	if status == 0 || !strings.Contains(out, "publication.worker_upgrade_required") {
		t.Fatalf("Runtime 51 accepted publication authority: [%d] %s", status, out)
	}
	refused, problem := store.RequestByIdempotencyKey(key + "-publication-refused")
	fatal(t, problem)
	if refused == nil {
		t.Fatal("publication refusal lost its client request")
	}
	authority, problem := store.MachinePublicationIntent(refused.ID)
	fatal(t, problem)
	link, problem = store.MachineExecution(refused.ID)
	fatal(t, problem)
	if len(authority) != 0 || len(link.Submission) != 0 || len(link.Receipt) != 0 {
		t.Fatal("old Runtime caused authority issuance or execution transmission")
	}
	status, out = runCozyPath(t, layout.Root, path, "run", "cancel", refused.ID, "--json")
	if status != 0 {
		t.Fatalf("cancel unsent publication refusal [%d]: %s", status, out)
	}
}
