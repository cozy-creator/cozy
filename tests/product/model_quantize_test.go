package producttest

import (
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
)

// quantizerJob is a package's quantizer for one lane: a model input and one weights output
// named for the lane.
func quantizerJob(lane string) string {
	return `{"models":[{"class":"Source","component_use":{"denoise":["model"]},"path":"` + lane + `.models.source"}],"name":"` + lane +
		`","publishes":false,"request":{"fields":[{"name":"source","type":{"union":["null",{"input":"model"}]},"wire":"optional"}]},` +
		`"result":{"input":"model"},"weights_outputs":[{"max_bytes":1048576,"mime_type":"application/vnd.cozy.model-manifest","output_id":"` + lane + `"}]}`
}

// installQuantizer records pkg as an installed published package with this interface.
func installQuantizer(t *testing.T, root, pkg string, iface []byte) {
	t.Helper()
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	dir := filepath.Join(root, "installs", strings.ReplaceAll(pkg, "/", "-"))
	must(t, os.MkdirAll(filepath.Dir(launch.PackageInterfacePath(dir)), 0700))
	must(t, os.WriteFile(launch.PackageInterfacePath(dir), iface, 0600))
	_, problem = store.Activate(records.PackageInstall{ID: filepath.Base(dir), Package: pkg, Major: 1,
		Version: "1.0.0", SourceKind: "tensorhub", Dir: dir, Platform: "linux-x86"})
	fatal(t, problem)
}

// `cozy model quantize <model> --fp8` is the serving package's quantizer run as an ordinary
// conversion: the job whose one weights output is named for the lane, bound to the exact
// checkpoint, uploading to the model's own repository.
func TestModelQuantizeRunsTheServingPackagesQuantizer(t *testing.T) {
	iface := []byte(`{"application":"q:app","entrypoints":[],"format":"cozy.package.interface/1","jobs":[` + quantizerJob("fp8") + `]}`)
	root, _, _, digest, _ := runModelCatalog(t, func(_ *http.ServeMux, detail *hub.PackageReleaseDetail) {
		contract, problem := launch.DecodePackageInterface(iface)
		fatal(t, problem)
		detail.PackageInterface = iface
		detail.Release.PackageInterfaceDigest = assessmentDigest(contract.Raw)
		detail.Release.PackageInterfaceLength = int64(len(iface))
	})
	for name, args := range map[string][]string{
		"name the lane to write":            {"proof/source"},
		"one run writes one lane":           {"proof/source", "--fp8", "--mxfp8"},
		"no installed package quantizes":    {"proof/source", "--fp8"},
		"proof/quantize 1.0.0 has no mxfp8": {"proof/source", "--mxfp8", "--package", "proof/quantize"},
		"weights output named mxfp8":        {"proof/source", "--mxfp8", "--package", "proof/quantize"},
		"is a local alias":                  {"local/source", "--fp8"},
	} {
		if code, out := runCozy(t, root, append([]string{"model", "quantize", "--rental-only"}, args...)...); code == 0 || !strings.Contains(out, name) {
			t.Fatalf("%v did not refuse with %q: %d %s", args, name, code, out)
		}
	}
	// An installed package from this hub whose quantizer the checkpoint's components satisfy
	// serves the model; one that reads a component the checkpoint lacks does not.
	installQuantizer(t, root, "proof/quantize", iface)
	installQuantizer(t, root, "proof/other", []byte(strings.ReplaceAll(string(iface), `["model"]`, `["unet"]`)))
	startDaemonProcess(t, root)
	for key, args := range map[string][]string{
		"quantize-name":       {"proof/source"},
		"quantize-lane":       {"proof/source@1.0.0/bf16", "proof/output"},
		"quantize-checkpoint": {"proof/source#" + digest, "--package", "proof/quantize"},
	} {
		code, out := runCozy(t, root, append([]string{"model", "quantize", "--fp8", "--rental-only", "--json", "--idempotency-key", key}, args...)...)
		if code != 0 {
			t.Fatalf("%v did not queue: %d %s", args, code, out)
		}
	}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	for key, destination := range map[string]string{"quantize-name": "proof/source", "quantize-lane": "proof/output", "quantize-checkpoint": "proof/source"} {
		row, problem := store.RequestByIdempotencyKey(key)
		fatal(t, problem)
		if row == nil || row.Package != "proof/quantize" || row.Entrypoint != "fp8" || len(row.Models) != 1 ||
			row.Models[0].Manifest != digest || row.ModelTransfer == nil || row.ModelTransfer.Destination != destination ||
			len(row.ModelTransfer.Outputs) != 1 || row.ModelTransfer.Outputs[0].Name != "fp8" {
			t.Fatalf("%s is not the fp8 conversion of the checkpoint into %s: %+v", key, destination, row)
		}
	}
}
