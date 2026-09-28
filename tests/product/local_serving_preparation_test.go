package producttest

import (
	"bytes"
	"testing"

	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
)

// A published install's preparation metadata retains the exact interface bytes the machine
// path sends; a digest alone cannot satisfy Runtime's preparation request.
func TestLocalPreparationRetainsPublishedInterface(t *testing.T) {
	iface, problem := launch.DecodePackageInterface([]byte(`{"format":"cozy.package.interface/1","application":"proof:app","entrypoints":[],"jobs":[]}`))
	fatal(t, problem)
	facts := launch.Facts{Install: records.PackageInstall{ID: "published", Package: "proof/model",
		Version: "1.0.0", SourceKind: "tensorhub", Dir: t.TempDir()},
		PackageInterface: iface}
	spec := facts.PreparationSpec()
	if spec.Preparation == nil || !spec.Preparation.Published ||
		!bytes.Equal(spec.Preparation.PackageInterface, iface.Raw) {
		t.Fatalf("local launch lost published interface: %+v", spec.Preparation)
	}
}
