package producttest

import (
	"bytes"
	"context"
	"io"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/home"
)

// Inspection remains read-only while the live daemon owns the root, but a custody
// writer must acquire that same lock before it may touch an operation.
func TestSourceCustodyCannotRunBesideTheDaemon(t *testing.T) {
	root := t.TempDir()
	cfg := config.Config{Home: root, HubURL: "http://127.0.0.1:1"}
	layout, problem := home.Open(root)
	fatal(t, problem)
	held, problem := daemon.Hold(layout, "127.0.0.1:1", "")
	fatal(t, problem)
	defer held.Release()
	_, problem = cli.SyncStoredSourceCustody(context.Background(), cfg, "existing-job", "existing-rental", "existing-boot", io.Discard, nil, nil)
	if problem == nil || !strings.Contains(problem.Message, "another Cozy daemon") {
		t.Fatal("custody writer did not respect the existing owner lock")
	}
	held.Release()
	second, problem := daemon.Hold(layout, "", "")
	fatal(t, problem)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	command := exec.CommandContext(ctx, cozyBin, "up", "--json", "--full")
	// This `up` is expected to refuse, but the root is registered for reaping anyway:
	// the suite's guarantee is about what a child COULD start, not what it should.
	trackDaemonRoot(t, root)
	command.Env = cfg.Child("COZY_HOME="+root, "TENSORFS_HOME="+filepath.Join(root, "tensorfs"), "TENSORHUB_URL=http://127.0.0.1:1")
	output, err := command.CombinedOutput()
	if ctx.Err() != nil || err == nil || !bytes.Contains(output, []byte("daemon.operator_owned")) {
		t.Fatalf("cozy up did not refuse the operator's live root: %v %s", err, output)
	}
	second.Release()
}
