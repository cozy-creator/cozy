package live

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/launch"
	"github.com/cozy-creator/cozy-creator/internal/records"
)

// TestRuntimeMetadataQueriesAreBounded drives a real child process. Bindings and the job
// describe query must stop at their metadata deadline; host inspection remains unbounded
// by that deadline because fit/doctor may legitimately inspect a real device.
func TestRuntimeMetadataQueriesAreBounded(t *testing.T) {
	if launch.DefaultRuntimeQueryTimeout <= 0 {
		t.Fatal("the production runtime metadata-query deadline is disabled")
	}
	dir := t.TempDir()
	bin := filepath.Join(dir, "cozy-runtime") //cozy:allow the LIVE arm supplies a sleeping independent runtime executable to prove the metadata deadline
	script := `#!/bin/sh
case "$4" in
  bindings|describe) exec sleep 30 ;;
  doctor) sleep 0.15; printf '{}\n' ;;
  fit) sleep 0.15; printf '{"host":{},"verdicts":[]}\n' ;;
esac
`
	if err := os.WriteFile(bin, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	runtime := launch.RuntimeCLI{ //cozy:allow the LIVE arm enters the one launch adapter directly so it can inject a 50ms query deadline
		Bin: bin, Dir: dir, Home: dir, Env: []string{"PATH=/usr/bin:/bin"},
		QueryTimeout: 50 * time.Millisecond,
	}
	assertStalled := func(t *testing.T, e *exit.Error) {
		t.Helper()
		if e == nil || e.Code != exit.Deadline || e.ErrName() != "runtime_query_stalled" {
			t.Fatalf("metadata query did not return runtime_query_stalled/deadline: %s", briefly(e))
		}
	}

	t.Run("bindings", func(t *testing.T) {
		_, _, e := runtime.Bindings()
		assertStalled(t, e)
	})
	t.Run("job describe", func(t *testing.T) {
		facts := launch.Facts{
			Install:           records.PackageInstall{Package: "fake/slow"},
			PackageDescriptor: &launch.PackageDescriptor{Jobs: []launch.Entrypoint{{Name: "slow"}}},
			RuntimeCLI:        runtime,
		}
		_, e := facts.Job("slow")
		assertStalled(t, e)
	})
	t.Run("host inspection is not metadata bounded", func(t *testing.T) {
		if _, e := runtime.HostFacts(); e != nil {
			t.Fatalf("doctor inherited the metadata deadline: %s", briefly(e))
		}
		if _, _, e := runtime.Fit("", nil); e != nil {
			t.Fatalf("fit inherited the metadata deadline: %s", briefly(e))
		}
	})
}
