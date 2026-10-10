package producttest

import (
	"bufio"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/machines"
)

// diskWaits are the system calls a process made that wait on the disk's journal, where a
// read does not: `commits` are syncs of the records log (one per synced transaction), `syncs`
// every other one (a checkpoint of that log among them) and `changes` metadata writes (chmod,
// rename, unlink, mkdir, truncate).
type diskWaits struct {
	commits, syncs, changes int
	calls                   []string
}

// traced reads the disk waits a strace of path recorded in [from, to].
func traced(t *testing.T, path string, from, to time.Time) diskWaits {
	t.Helper()
	file, err := os.Open(path)
	must(t, err)
	defer file.Close()
	var waits diskWaits
	opened := map[string]string{}  // descriptor -> the file it last opened
	opening := map[string]string{} // thread -> the file an open it has not finished names
	lines := bufio.NewScanner(file)
	lines.Buffer(make([]byte, 1<<20), 1<<20)
	for lines.Scan() {
		fields := strings.SplitN(lines.Text(), " ", 3)
		if len(fields) < 3 {
			continue
		}
		descriptor := strings.TrimSpace(fields[2][strings.LastIndex(fields[2], "=")+1:])
		if strings.HasPrefix(fields[2], "<... open") {
			opened[descriptor] = opening[fields[0]]
			continue
		}
		name, rest, called := strings.Cut(fields[2], "(")
		if !called || strings.Contains(rest, "= -1 ") {
			continue
		}
		if name == "openat" || name == "open" {
			if quoted := strings.SplitN(rest, `"`, 3); len(quoted) == 3 {
				opened[descriptor], opening[fields[0]] = quoted[1], quoted[1]
			}
			continue
		}
		at, err := strconv.ParseFloat(fields[1], 64)
		if err != nil || at < float64(from.UnixNano())/1e9 || at > float64(to.UnixNano())/1e9 {
			continue
		}
		switch target := opened[strings.TrimRight(strings.Fields(rest)[0], ")")]; {
		case name != "fsync" && name != "fdatasync":
			waits.changes++
		case strings.HasSuffix(target, "creator.sqlite-wal"):
			waits.commits++
		default:
			waits.syncs++
			fields[2] += " " + target
		}
		waits.calls = append(waits.calls, fields[2])
	}
	return waits
}

const tracedCalls = "trace=open,openat,fsync,fdatasync,chmod,fchmod,fchmodat,rename,renameat,renameat2,unlink,unlinkat,mkdir,mkdirat,ftruncate"

// A warm run waits on the disk only where a record must outlive a crash: the request with its
// placement, its dispatch, its acceptance and its outcome. The command only reads, so it
// changes nothing on disk. On a busy disk each such wait is seconds (run 5323: 7.4 s before the
// command's first line and 10 s inside Submit, where rewriting an unchanged file mode measured
// up to 2.8 s and an fsync 2.6 s).
func TestAWarmRunWaitsOnFewDiskWrites(t *testing.T) {
	strace, err := exec.LookPath("strace")
	if err != nil {
		t.Skip("strace counts the disk waits")
	}
	h, root, _, _ := parityMachines(t)
	resolved := seedProbe(t, h, root, machines.Local, "tessa")
	publishParityRelease(t, h, root, probeProject(t, "proof/probe@1.0.0/bf16"))
	binding := `{"bindings":[{"slot":"touch.models.source","model":"proof/probe","release":"1.0.0","revision":1,"ladder":[{"gpu":"*","lane":"bf16"}]}]}`
	account, probe := h.server.Config.Handler, probeModel(t, resolved)
	h.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/v1/packages/"+parityPublished+"/bindings":
			_, _ = w.Write([]byte(binding))
		case probe(w, r):
		default:
			account.ServeHTTP(w, r)
		}
	})
	touch := func(args ...string) []string {
		return append([]string{"run", parityPublished + "/touch", "value=1", "--await", "--json"}, args...)
	}
	for _, venue := range [][]string{nil, {"--rental=tessa"}} {
		if code, out := runCozy(t, root, touch(venue...)...); code != 0 {
			t.Fatalf("cold touch %v [exit %d]\n%s", venue, code, out)
		}
	}

	// The daemon runs under strace from here: a symlink named as its process is.
	if code, out := runCozy(t, root, "down"); code != 0 {
		t.Fatalf("down [exit %d]\n%s", code, out)
	}
	entry := filepath.Join(t.TempDir(), "cozy-daemon")
	must(t, os.Symlink(cozyBin, entry))
	daemonTrace := filepath.Join(root, "daemon.strace")
	served := exec.Command(strace, "-f", "-ttt", "-e", tracedCalls, "-o", daemonTrace, entry)
	served.Env = childEnv(t, root)
	must(t, served.Start())
	t.Cleanup(func() { _, _ = runCozy(t, root, "down"); _ = served.Wait() })
	eventually(t, root, "the traced daemon", func() bool { return daemon.Probe(config.Config{Home: root}).Up })

	for _, venue := range []struct {
		name string
		args []string
	}{{"local", nil}, {"rental", []string{"--rental=tessa"}}} {
		// Once more untraced: this daemon's first run reattaches its machines.
		if code, out := runCozy(t, root, touch(venue.args...)...); code != 0 {
			t.Fatalf("touch on %s [exit %d]\n%s", venue.name, code, out)
		}
		cliTrace := filepath.Join(root, venue.name+".strace")
		command := exec.Command(strace, append([]string{"-f", "-ttt", "-e", tracedCalls, "-o", cliTrace, cozyBin}, touch(venue.args...)...)...)
		command.Env = childEnv(t, root)
		began := time.Now()
		out, err := command.CombinedOutput()
		ended := time.Now()
		if err != nil || !strings.Contains(string(out), `"value":2`) {
			t.Fatalf("traced touch on %s: %v\n%s", venue.name, err, out)
		}
		cli, served := traced(t, cliTrace, began, ended), traced(t, daemonTrace, began, ended)
		t.Logf("warm run on %s: command %d disk waits; daemon %d commits, %d other syncs, %d changes\n  %s", venue.name,
			len(cli.calls), served.commits, served.syncs, served.changes, strings.Join(served.calls, "\n  "))
		if len(cli.calls) > 0 {
			t.Errorf("the command of a warm run on %s waited on the disk %d times; it only reads:\n  %s", venue.name, len(cli.calls), strings.Join(cli.calls, "\n  "))
		}
		if served.commits > 4 || served.changes > 0 {
			t.Errorf("the daemon made %d synced commits and %d metadata writes for a warm run on %s; its request (placed), dispatch, acceptance and outcome are 4 commits",
				served.commits, served.changes, venue.name)
		}
	}

	// Holding a rental and running nothing, the daemon re-reads its rentals at the Hub
	// (every 10 s) and writes what it learns without waiting on the disk.
	began := time.Now()
	time.Sleep(12 * time.Second)
	if idle := traced(t, daemonTrace, began, time.Now()); len(idle.calls) > 0 {
		t.Errorf("the idle daemon waited on the disk %d times:\n  %s", len(idle.calls), strings.Join(idle.calls, "\n  "))
	}
}
