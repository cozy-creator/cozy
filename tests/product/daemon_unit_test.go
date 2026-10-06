package producttest

import (
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/calcifer"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/userunit"
)

// A daemon a command autostarts runs as its own user unit, never in the caller's scope: ending
// the session scope the command ran in (an agent's, a terminal's) leaves it serving, `cozy
// down` ends the unit, and the next command brings it back as that unit.
func TestTheDaemonOutlivesTheScopeThatStartedIt(t *testing.T) {
	if !userunit.Available() {
		t.Skip("requires a systemd user manager")
	}
	root := t.TempDir()
	unit := userunit.Name("calcifer", root, false)
	t.Cleanup(func() { _ = userunit.Stop(unit) })
	scope := "cozy-test-" + strings.ToLower(randomToken(t)[:8])
	up := exec.Command("systemd-run", "--user", "--scope", "--quiet", "--unit="+scope, "--", cozyBin, "up", "--json")
	up.Env = childEnv(t, root)
	if out, err := up.CombinedOutput(); err != nil {
		t.Fatalf("cozy up in a scope: %v\n%s", err, out)
	}
	// The scope ends as a session does: everything left in it is killed.
	_ = exec.Command("systemctl", "--user", "stop", scope+".scope").Run()
	time.Sleep(time.Second)
	if !calcifer.Probe(config.Config{Home: root}).Up || userunit.Property(unit, "ActiveState") != "active" {
		t.Fatalf("the daemon did not outlive the scope that started it (unit %s is %q)", unit, userunit.Property(unit, "ActiveState"))
	}
	if code, out := runCozy(t, root, "down", "--json"); code != 0 {
		t.Fatalf("cozy down [exit %d]: %s", code, out)
	}
	for deadline := time.Now().Add(time.Minute); userunit.Property(unit, "ActiveState") == "active"; time.Sleep(100 * time.Millisecond) {
		if time.Now().After(deadline) {
			t.Fatalf("cozy down left the daemon's unit %s running", unit)
		}
	}
	if code, out := runCozy(t, root, "up", "--json"); code != 0 {
		t.Fatalf("cozy up after down [exit %d]: %s", code, out)
	}
	pid := userunit.MainPID(unit)
	argv, _ := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/cmdline")
	if pid == 0 || filepath.Base(strings.SplitN(string(argv), "\x00", 2)[0]) != "calcifer" {
		t.Fatalf("the next command did not bring the daemon back as unit %s (pid %d: %q)", unit, pid, argv)
	}
}
