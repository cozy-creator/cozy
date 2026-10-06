package producttest

import (
	"database/sql"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/records"
)

// A settled worker.v1 run that still carries a cancel is history: the daemon resumes nothing
// for it. Restarting its observer at once made a busy loop that slowed `cozy run list` 3×.
func TestSettledArchivedRunWithACancelLeavesTheDaemonIdle(t *testing.T) {
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: http://127.0.0.1:1\ntensorhub_token: unreachable\n"), 0o600))
	path := filepath.Join(root, "creator.sqlite")
	store, problem := records.Open(path)
	fatal(t, problem)
	request, _, problem := store.Submit(records.Request{ID: "archive-run", IdemKey: "saved-worker",
		Package: "local/archived", Entrypoint: "main", Kind: "job", Payload: []byte(`{}`),
		BodyDigest: childDigest("1"), RetainWork: true, MachineExecutionObserver: true})
	fatal(t, problem)
	fatal(t, store.LinkMachineExecution(request.ID, "local"))
	store.Close()
	archived := map[string][]byte{}
	for _, name := range []string{"submission", "receipt", "outcome"} {
		data, err := os.ReadFile("testdata/record-archive/" + name + ".bin")
		must(t, err)
		archived[name] = data
	}
	db, err := sql.Open("sqlite", path)
	must(t, err)
	_, err = db.Exec(`UPDATE machine_executions SET submission=?,receipt=?,outcome=?,pending_control=?,cancel_requested=1,collected=1 WHERE request_id=?`,
		archived["submission"], archived["receipt"], archived["outcome"], []byte("cancel"), request.ID)
	must(t, err)
	_, err = db.Exec(`UPDATE requests SET state='canceled' WHERE id=?`, request.ID)
	must(t, err)
	must(t, db.Close())

	daemon := startDaemonProcess(t, root)
	time.Sleep(time.Second)
	before, began := cpuTime(t, daemon.cmd.Process.Pid), time.Now()
	time.Sleep(3 * time.Second)
	busy, elapsed := cpuTime(t, daemon.cmd.Process.Pid)-before, time.Since(began)
	if busy > elapsed/10 {
		t.Fatalf("the daemon used %s of CPU in %s with nothing to do\n%s", busy, elapsed, tail(filepath.Join(root, "daemon.log")))
	}
	t.Logf("the daemon used %s of CPU in %s", busy, elapsed)
}

// cpuTime is a process's user and system CPU time (USER_HZ is 100 on Linux).
func cpuTime(t *testing.T, pid int) time.Duration {
	t.Helper()
	stat, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/stat")
	must(t, err)
	fields := strings.Fields(string(stat[strings.LastIndexByte(string(stat), ')')+1:]))
	user, err := strconv.ParseInt(fields[11], 10, 64)
	must(t, err)
	system, err := strconv.ParseInt(fields[12], 10, 64)
	must(t, err)
	return time.Duration(user+system) * 10 * time.Millisecond
}
