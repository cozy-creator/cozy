package producttest

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/tfs"
)

func TestTensorFSRefusalSurvivesAnEarlierNamespaceWarning(t *testing.T) {
	executable := filepath.Join(t.TempDir(), "tfs")
	// Independent subprocess output, exercising the real tool caller and error envelope.
	must(t, os.WriteFile(executable, []byte("#!/bin/sh\nprintf '%s\\n' 'unshare: uid_map: Operation not permitted' 'REFUSED GOLDEN_MISMATCH: golden suite absent' 'worker exited Some(1)' >&2\nexit 1\n"), 0700))
	tool := tfs.Tool{Bin: executable, Root: t.TempDir()}
	_, problem := tool.RunSource(context.Background(), "source-plan.json")
	if problem == nil || !strings.Contains(problem.Message, "REFUSED GOLDEN_MISMATCH: golden suite absent") || strings.Contains(problem.Message, "uid_map") {
		t.Fatalf("reported a probe warning instead of refusal: %v", problem)
	}
}
