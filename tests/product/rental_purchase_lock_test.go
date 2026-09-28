package producttest

import (
	"bytes"
	"context"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// runCozyWithin runs one CLI verb under the test's own deadline; a verb still running
// at it is killed and reported as hung, never waited on.
func runCozyWithin(t *testing.T, root string, args ...string) (int, string, string) {
	t.Helper()
	const within = 20 * time.Second
	ctx, cancel := context.WithTimeout(context.Background(), within)
	defer cancel()
	cmd := exec.CommandContext(ctx, "/usr/bin/nice", append([]string{"-n", "19", cozyBin}, args...)...)
	cmd.Env = childEnv(t, root)
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	_ = cmd.Run()
	if ctx.Err() != nil {
		t.Fatalf("cozy %s hung past %s\n%s", strings.Join(args, " "), within,
			tail(filepath.Join(root, "daemon.log")))
	}
	return cmd.ProcessState.ExitCode(), stdout.String(), stderr.String()
}
