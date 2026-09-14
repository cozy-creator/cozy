package producttest

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestPrivateHostNativeRootBytesSurviveClientExit(t *testing.T) {
	if *privateChildRuntimeWheel == "" {
		t.Skip("requires exact Runtime wire55 input wheel")
	}
	layout, store, host, path, _ := startActualChildHost(t)
	wheel, err := filepath.Abs(*privateChildRuntimeWheel)
	must(t, err)
	script, source, report, data := rootByteInputProject(t, wheel)
	key := fmt.Sprintf("root-input-%d", time.Now().UnixNano())
	code, out := runCozyPath(t, layout.Root, path, "run", script, "tree=original", "--input-tree", "original="+source, "--asset", "report="+report, "--rental", "child-host", "--idempotency-key", key, "--json")
	if code != 0 || !strings.Contains(out, `"machine_accepted":true`) {
		t.Fatalf("private native input intake [%d]: %s", code, out)
	}
	request, problem := store.RequestByIdempotencyKey(key)
	fatal(t, problem)
	if request == nil {
		t.Fatal("private root input request absent")
	}
	must(t, os.WriteFile(report, []byte("changed original file"), 0600))
	must(t, os.WriteFile(filepath.Join(source, "report.json"), []byte("changed original tree"), 0600))
	must(t, os.Remove(filepath.Join(source, "extra.txt")))
	if code, out := runCozyPath(t, layout.Root, path, "down", "--json"); code != 0 || daemonOnRoot(layout.Root) != 0 {
		t.Fatalf("private input client detach [%d]: %s", code, out)
	}
	probe := func() []byte {
		t.Helper()
		data, err := exec.Command("docker", "exec", host.Container, "python3", "-c", `import json,sqlite3,sys
from cozy_runtime.internal.placement_materialization import LOCAL_TENSORFS_ROOT
c=sqlite3.connect('file:'+str(LOCAL_TENSORFS_ROOT/'.cozy-workspace/journal.sqlite3')+'?mode=ro',uri=True,timeout=10)
c.row_factory=sqlite3.Row
request=sys.argv[1]
value=dict(c.execute('select request,state,collected from executions where request=?',(request,)).fetchone())
value['intakes']=c.execute("select count(*) from input_tree_intakes where request=? and state='released'",(request,)).fetchone()[0]
value['children']=c.execute('select count(*) from execution_calls where parent_request=?',(request,)).fetchone()[0]
value['attempts']=c.execute('select count(*) from attempts where request=? or request in(select child_request from execution_calls where parent_request=?)',(request,request)).fetchone()[0]
print(json.dumps(value))`, request.ID).CombinedOutput()
		if err != nil {
			t.Fatalf("read actual input journal: %v %s", err, data)
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
		if daemonOnRoot(layout.Root) != 0 {
			t.Fatal("client returned before private Runtime completion")
		}
		if state.State == "succeeded" && state.Collected == 0 {
			break
		}
		if state.State == "failed" || state.State == "canceled" || time.Now().After(deadline) {
			t.Fatalf("private Runtime did not finish independently: %s", raw)
		}
		time.Sleep(250 * time.Millisecond)
	}
	if code, out := runCozyPath(t, layout.Root, path, "run", "watch", request.ID, "--json"); code != 0 {
		t.Fatalf("collect private native input result [%d]: %s", code, out)
	}
	exports, problem := store.OutputExportOf(request.ID)
	fatal(t, problem)
	if exports == nil || exports.State != "published" || len(exports.PublishedPaths) != 1 {
		t.Fatal("private report bundle is not exported")
	}
	actual, err := os.ReadFile(filepath.Join(exports.PublishedPaths[0], "report.json"))
	must(t, err)
	if string(actual) != string(data) {
		t.Fatal("private input followed an edited source path")
	}
	attempts, problem := store.Attempts(request.ID)
	fatal(t, problem)
	children, problem := store.Children(request.ID)
	fatal(t, problem)
	if len(attempts) != 0 || len(children) != 0 {
		t.Fatal("private input intake created Creator execution rows")
	}
	raw := probe()
	var final struct {
		State                                  string
		Collected, Intakes, Children, Attempts int
	}
	must(t, json.Unmarshal(raw, &final))
	if final.State != "succeeded" || final.Collected != 1 || final.Intakes != 2 || final.Children != 1 || final.Attempts != 2 {
		t.Fatalf("private input custody or execution set differs: %s", raw)
	}
	processes, err := exec.Command("docker", "exec", host.Container, "python3", "-c", `import json
from pathlib import Path
found=[]
for path in Path('/proc').glob('[0-9]*/cmdline'):
 try: found.append(path.read_bytes().decode().split('\0')[:2])
 except (OSError,ProcessLookupError): pass
print(json.dumps(found))`).CombinedOutput()
	must(t, err)
	var commands [][]string
	must(t, json.Unmarshal(processes, &commands))
	for _, command := range commands {
		for _, argument := range command {
			if argument == "cozy" || argument == "cozy-daemon" || strings.HasSuffix(argument, "/cozy") || strings.HasSuffix(argument, "/cozy-daemon") {
				t.Fatal("Creator process appeared inside the private Host")
			}
		}
	}
	must(t, os.WriteFile(filepath.Join(layout.Root, "native-root-input-proof.json"), raw, 0600))
	t.Logf("actual private Host %s; input source edited/client absent; %s", host.Container, raw)
}
