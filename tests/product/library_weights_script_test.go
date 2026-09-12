package producttest

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
)

// A real native Model input and WeightsSink stay in this attempt even when its
// ordinary helper library also exports a managed callable. No worker is mocked.
func TestPrivateLibraryScriptKeepsModelAndWeightsInItsOwnAttempt(t *testing.T) {
	wheel := *privateChildRuntimeWheel
	if wheel == "" {
		t.Skip("requires the exact candidate Runtime wheel")
	}
	root, err := os.MkdirTemp("", "cozy-library-weights-")
	must(t, err)
	control := filepath.Join(t.TempDir(), "control")
	uv := func(args ...string) {
		t.Helper()
		if output, err := exec.Command("uv", args...).CombinedOutput(); err != nil {
			t.Fatalf("uv: %v %s", err, output)
		}
	}
	uv("venv", control, "--python", "3.12")
	deps := []string{"pip", "install", "--python", filepath.Join(control, "bin", "python"), wheel}
	if *privateChildTensorFSWheel != "" {
		deps = append(deps, *privateChildTensorFSWheel)
	}
	uv(deps...)
	path := filepath.Join(control, "bin")
	for _, item := range childEnv(t, root) {
		if strings.HasPrefix(item, "PATH=") {
			path += string(os.PathListSeparator) + strings.TrimPrefix(item, "PATH=")
		}
	}
	t.Cleanup(func() {
		compositionDown(t, root, path)
		if t.Failed() {
			t.Log("library script evidence retained", root)
		} else {
			_ = os.RemoveAll(root)
		}
	})
	project := copyPrivateTensorProject(t, root, "")
	// Catalog model preparation currently also reads construction config, even
	// though this job never constructs a model. Include a real inline config so
	// the actual Runtime preparation can validate the captured slot contract.
	producerSource := filepath.Join(project, "source", "tensor_source.py")
	sourceCode, err := os.ReadFile(producerSource)
	must(t, err)
	source := strings.Replace(string(sourceCode), "App, Context,", "App, Context, WeightsConfig,", 1)
	source = strings.Replace(source, "configs={},", `configs={"model": WeightsConfig(data=b"{}")},`, 1)
	source = strings.Replace(source, "        writer.add_part(", "        writer.add_config(\"model\", b\"{}\")\n        writer.add_part(", 1)
	must(t, os.WriteFile(producerSource, []byte(source), 0600))
	script := filepath.Join(project, "recipe.py")
	body, err := os.ReadFile(script)
	must(t, err)
	seed := strings.Replace(string(body), "    await candidate(source=original, factor=0)", "    return original", 1)
	must(t, os.WriteFile(script, []byte(seed), 0600))
	if status, output := runCozyPath(t, root, path, "run", script, "--await", "--json"); status != 0 {
		t.Fatalf("native fixture seed failed [%d]: %s", status, output)
	}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	producer, problem := store.RequestByReference("2")
	fatal(t, problem)
	if producer == nil || producer.State != "succeeded" {
		t.Fatalf("native seed did not complete: %+v", producer)
	}
	weights, problem := store.AllModelTransferWeights(producer.ID, producer.Ordinal)
	fatal(t, problem)
	if len(weights) != 1 {
		t.Fatalf("native seed has no committed manifest: %+v", weights)
	}
	// Register the already-produced bytes as an ordinary local checkpoint. This
	// uses native repository custody, not a fake Model loader or HTTP response.
	register := exec.Command("tfs", "local", "replace", filepath.Join(root, "tensorfs"), "library-seed", strings.TrimPrefix(weights[0].ManifestID, "sha256:"), weights[0].ManifestID, strconv.FormatInt(weights[0].ManifestLength, 10), "--observed", "absent") //cozy:allow fixture uses native repository CLI to register a real produced checkpoint
	if output, err := register.CombinedOutput(); err != nil {
		t.Fatalf("retain native fixture checkpoint: %v %s", err, output)
	}
	library := filepath.Join(project, "candidate", "tensor_candidate.py")
	helper, err := os.ReadFile(library)
	must(t, err)
	helper = append(helper, []byte("\ndef scale_values(values):\n    return (value * 2 for value in values)\n")...)
	must(t, os.WriteFile(library, helper, 0600))
	metadata, _, _ := strings.Cut(string(body), "# ///\nfrom")
	metadata = strings.Replace(metadata, "# [tool.uv.sources]", "# [tool.cozy.weights]\n# weights = 4096\n# [tool.uv.sources]", 1)
	code := metadata + `# ///
import struct
from cozy_runtime.author import ModelArtifact, WeightsPart, WeightsSink, WeightsTarget, WeightsTensor
from tensor_candidate import Source, PLAIN, scale_values

def main(*, source: Source, artifacts: WeightsSink) -> ModelArtifact:
    tensor = WeightsTensor(logical_dtype="f32", shape=(512,), encoding=PLAIN,
                          parts={"value": WeightsPart("f32", (512,))})
    with artifacts.open("weights", sources={"original": source},
        targets={"model": WeightsTarget(source="original", source_component="model",
            drop=("weight",), add={"weight": tensor})},
        configs={}, order=(("model", "weight"),)) as writer:
        data = bytearray(2048)
        writer.source_read_into("original", "model", "weight", "value", 0, data)
        writer.add_part("model", "weight", "value",
                        struct.pack("<512f", *scale_values(struct.unpack("<512f", data))))
        return writer.commit().artifact
`
	must(t, os.WriteFile(script, []byte(code), 0600))
	status, output := runCozyPath(t, root, path, "run", script, "model.source=local/library-seed#"+weights[0].ManifestID, "--await", "--json")
	if status != 0 {
		t.Fatalf("ordinary library script failed [%d]: %s", status, output)
	}
	request, problem := store.RequestByReference("3")
	fatal(t, problem)
	if request == nil || request.State != "succeeded" || len(request.Models) != 1 || request.WeightsOutputs == "" {
		t.Fatalf("ordinary model/weights attempt missing: %+v", request)
	}
	installed, problem := store.Install(request.InstallID)
	fatal(t, problem)
	if installed == nil {
		t.Fatal("native output request lost its captured interface")
	}
	selection, err := json.Marshal(request.Models)
	must(t, err)
	selectionPath := filepath.Join(root, "model-selection.json")
	must(t, os.WriteFile(selectionPath, selection, 0600))
	checkSlots := exec.Command(filepath.Join(control, "bin", "python"), filepath.Join("testdata", "verify_job_model_slots.py"),
		launch.PackageInterfacePath(installed.Dir), selectionPath, filepath.Join(root, "tensorfs"))
	if output, err := checkSlots.CombinedOutput(); err != nil {
		t.Fatalf("captured job model selection cannot prepare on actual Runtime: %v %s", err, output)
	}
	children, problem := store.Children(request.ID)
	fatal(t, problem)
	if len(children) != 0 {
		t.Fatalf("helper import produced child requests: %+v", children)
	}
	outputs, problem := store.AllModelTransferWeights(request.ID, request.Ordinal)
	fatal(t, problem)
	if len(outputs) != 1 {
		t.Fatalf("ordinary native output missing: %+v", outputs)
	}
	read := exec.Command(filepath.Join(control, "bin", "python"), filepath.Join("testdata", "private_child_read.py"), outputs[0].ManifestID, "14", filepath.Join(root, "tensorfs"))
	if output, err := read.CombinedOutput(); err != nil {
		t.Fatalf("real transformed tensor: %v %s", err, output)
	}
	t.Log(fmt.Sprintf("one ordinary request read native Model and wrote value 14; %s", outputs[0].ManifestID))
}
