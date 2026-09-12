package producttest

import (
	"net/http"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
)

// Exercise CLI -> daemon admission using the existing metadata-only catalog.
// Its acquisition route refuses all rentals; this proves no model execution.
func TestRunServingAcceptsRetainedCheckpointWithoutRelease(t *testing.T) {
	root, _, _, digest, manifest := runModelCatalog(t, func(_ *http.ServeMux, detail *hub.PackageReleaseDetail) {
		iface := []byte(`{"application":"q:app","entrypoints":[{"models":[{"class":"Source","component_use":{},"path":"generate.models.model"}],"name":"generate","request":{"fields":[{"name":"prompt","type":"str"}]},"result":{"fields":[]}}],"format":"cozy.package.interface/1","jobs":[]}`)
		contract, problem := launch.DecodePackageInterface(iface)
		fatal(t, problem)
		detail.PackageInterface = iface
		detail.Release.PackageInterfaceDigest = contract.Digest
		detail.Release.PackageInterfaceLength = int64(len(iface))
	})
	startDaemonProcess(t, root)
	code, out := runCozy(t, root, "run", "proof/quantize/generate", "prompt=Compare the candidate",
		"model.model=proof/source#"+digest, "--rental-only", "--json", "--idempotency-key", "retained-serving")
	if code != 0 {
		t.Fatalf("exact candidate was rejected before serving admission: %d %s", code, out)
	}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	row, problem := store.RequestByIdempotencyKey("retained-serving")
	fatal(t, problem)
	if row == nil || row.IsJob() || len(row.Models) != 1 {
		t.Fatalf("candidate did not create one ordinary serving request: %+v", row)
	}
	model := row.Models[0]
	if !model.HubCheckpoint || !model.Downloadable() || model.Manifest != digest ||
		model.ManifestLength != int64(len(manifest)) || model.Release != "" || model.Lane != "" {
		t.Fatalf("serving admission changed the exact candidate: %+v", model)
	}
}
