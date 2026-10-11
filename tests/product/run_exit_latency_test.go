package producttest

import (
	"database/sql"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sync"
	"testing"
	"time"
)

// A run's command exits within a second of its outcome reaching this computer, plus the time
// its files take to arrive, alone or beside runs still going: nothing after an outcome waits on
// a poll, a timer, another run's fetch, a second connection or a sync of telemetry. How long
// the machine took to deliver the outcome after its journal's finish is the machine's own, and
// is logged.
func TestRunExitsPromptlyAfterItsOutcome(t *testing.T) {
	if *machineHostBinary == "" {
		t.Skip("requires -machine-host=<cozy-machine>")
	}
	root, err := os.MkdirTemp(os.TempDir(), "czx")
	must(t, err)
	provisionMachine(t, root)
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "machine", "stop")
		_, _ = runCozy(t, root, "down")
		if t.Failed() {
			t.Logf("evidence retained at %s", root)
		} else {
			_ = removeAllForce(root)
		}
	})
	if code, out := runCozy(t, root, "package", "install", weightlessProject(t)); code != 0 {
		t.Fatalf("package install [exit %d]\n%s", code, out)
	}
	// One run's tail: its machine's outcome (the journal's finish) to its command's exit.
	type finished struct {
		label  string
		id     string
		exited time.Time
	}
	run := func(label string, args ...string) finished {
		command := exec.Command("/usr/bin/nice", append([]string{"-n", "19", cozyBin, "run", localWeightlessRef + "/tile",
			"--await", "--json", "--out", t.TempDir()}, args...)...)
		command.Env = childEnv(t, root)
		output, err := command.CombinedOutput()
		exited := time.Now()
		id := regexp.MustCompile(`"request_id":"([^"]+)"`).FindSubmatch(output)
		if err != nil || id == nil {
			t.Fatalf("%s: %v\n%s", label, err, output)
		}
		return finished{label, string(id[1]), exited}
	}
	tail := func(f finished) time.Duration {
		t.Helper()
		open := func(path string) *sql.DB {
			db, err := sql.Open("sqlite", "file:"+path+"?mode=ro")
			must(t, err)
			return db
		}
		journal := open(filepath.Join(root, "machine", "root", "var", "lib", "cozy", "rust-machine", "execution", "executions.sqlite3"))
		defer journal.Close()
		creator := open(filepath.Join(root, "creator.sqlite"))
		defer creator.Close()
		var finished, observed int64
		must(t, journal.QueryRow(`SELECT json_extract(record,'$.finished_at_ms') FROM executions
			WHERE json_extract(record,'$.submission.request_id')=?`, f.id).Scan(&finished))
		must(t, creator.QueryRow(`SELECT json_extract(payload,'$.observed_unix_ms') FROM request_events
			WHERE request_id=? AND type='client.machine_work_finished'`, f.id).Scan(&observed))
		took := f.exited.Sub(time.UnixMilli(observed))
		t.Logf("%s: the machine delivered its outcome in %.3fs; outcome -> exit %.3fs", f.label,
			time.UnixMilli(observed).Sub(time.UnixMilli(finished)).Seconds(), took.Seconds())
		return took
	}
	run("cold", "size=32", "seed=1") // installs the environment and starts the machine
	for i := 2; i <= 3; i++ {
		if took := tail(run(fmt.Sprint("alone ", i), "size=32", fmt.Sprintf("seed=%d", i))); took > time.Second {
			t.Errorf("a run alone exited %.3fs after its outcome", took.Seconds())
		}
	}
	// Short runs that end while a long one goes on exit as promptly.
	var group sync.WaitGroup
	results := make([]finished, 4)
	for i := range results {
		group.Add(1)
		go func(i int) {
			defer group.Done()
			args := []string{"size=32", fmt.Sprintf("seed=%d", 10+i)}
			if i == 0 {
				args = append(args, "delay_ms=5000")
			}
			results[i] = run(fmt.Sprint("beside ", i), args...)
		}(i)
	}
	group.Wait()
	for _, f := range results[1:] {
		if took := tail(f); took > time.Second {
			t.Errorf("%s exited %.3fs after its outcome while another run went on", f.label, took.Seconds())
		}
	}
	tail(results[0])
}
