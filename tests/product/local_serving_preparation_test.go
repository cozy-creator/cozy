package producttest

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// The local published path must retain the exact interface bytes, as the rental
// and machine paths do. A digest alone cannot satisfy Runtime's wire-61 request.
func TestLocalPreparationRetainsPublishedInterface(t *testing.T) {
	bin := t.TempDir()
	must(t, os.WriteFile(filepath.Join(bin, "cozy-runtime"), []byte(stubRuntime(t, hostruntime.ToolFloor, pb.WireMinor)), 0700)) //cozy:allow a version-only test peer
	t.Setenv("PATH", bin)
	iface, problem := launch.DecodePackageInterface([]byte(`{"format":"cozy.package.interface/1","application":"proof:app","entrypoints":[],"jobs":[]}`))
	fatal(t, problem)
	facts := launch.Facts{Install: records.PackageInstall{ID: "published", Package: "proof/model",
		Version: "1.0.0", SourceKind: "tensorhub", Dir: t.TempDir()},
		PackageInterface: iface}
	spec, problem := facts.PreparationSpec([]string{"0"})
	fatal(t, problem)
	if spec.Preparation == nil || !spec.Preparation.Published ||
		!bytes.Equal(spec.Preparation.PackageInterface, iface.Raw) {
		t.Fatalf("local launch lost published interface: %+v", spec.Preparation)
	}
}
