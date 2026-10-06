package producttest

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/records"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

// Runs held on remote machines the daemon cannot reach leave it idle: a run whose machine is
// lost is never dialed again, and an unreachable one is asked again only on new evidence,
// never on a timer. The owner's laptop redialed two gone pods forever at ~15% CPU.
func TestRunsOnUnreachableMachinesLeaveTheDaemonIdle(t *testing.T) {
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: http://127.0.0.1:1\ntensorhub_token: unreachable\n"), 0o600))
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	for i := range 200 {
		id, machine := fmt.Sprintf("remote-run-%03d", i), fmt.Sprintf("endpoint-%016x", i)
		_, _, problem := store.Submit(records.Request{ID: id, IdemKey: id, Package: "proof/remote", Entrypoint: "main",
			Kind: "job", Payload: []byte(`{}`), BodyDigest: childDigest("1"), MachineExecutionObserver: true})
		fatal(t, problem)
		fatal(t, store.LinkMachineExecution(id, machine))
		fatal(t, store.AcceptRunV1(id, &v1.RunState{Id: id, Number: uint64(i + 1), State: "running", Attempt: 1}))
		if i%2 == 0 {
			fatal(t, store.LoseMachineExecution(id, machine, "its rental ended before it finished"))
		}
	}
	store.Close()

	daemon := startDaemonProcess(t, root)
	time.Sleep(10 * time.Second)
	before, began := cpuTime(t, daemon.cmd.Process.Pid), time.Now()
	time.Sleep(5 * time.Second)
	busy, elapsed := cpuTime(t, daemon.cmd.Process.Pid)-before, time.Since(began)
	if busy > elapsed/50 {
		t.Fatalf("the daemon used %s of CPU in %s with nothing it can reach\n%s", busy, elapsed, tail(filepath.Join(root, "daemon.log")))
	}
	t.Logf("the daemon used %s of CPU in %s", busy, elapsed)
}
