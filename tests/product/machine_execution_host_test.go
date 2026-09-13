package producttest

import (
	"flag"
	"os/exec"
	"strings"
	"testing"
)

var machineExecutionScript = flag.String("machine-execution-script", "", "captured script with an exact coherent Runtime dependency for the actual Host proof")

func TestMachineExecutionActualHostRoot(t *testing.T) {
	if *machineExecutionScript == "" {
		t.Skip("requires an explicit captured script and actual Host cohort")
	}
	layout, store, host, path, _ := startActualChildHost(t)
	status, out := runCozyPath(t, layout.Root, path, "run", *machineExecutionScript,
		"--rental", "child-host", "--await", "--json", "--idempotency-key", "machine-host-root")
	if status != 0 {
		t.Fatalf("Runtime-owned private Host invocation [%d]: %s", status, out)
	}
	request, problem := store.RequestByIdempotencyKey("machine-host-root")
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
	processes, err := exec.Command("docker", "exec", host.Container, "python3", "-c", `from pathlib import Path
for path in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  argv=path.read_bytes().split(b'\0')
  print(repr(argv[:2]))
 except (OSError,ProcessLookupError): pass
`).CombinedOutput()
	must(t, err)
	if strings.Contains(string(processes), "/opt/cozy/bin/cozy") || strings.Contains(string(processes), "b'cozy', b'daemon'") {
		t.Fatalf("Creator process is running inside private Host: %s", processes)
	}
	t.Logf("private Runtime root %s accepted and collected; no Creator attempts or children; %s", request.ID, out)
}
