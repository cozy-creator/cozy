package producttest

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

// A real native Model input and TensorFS output stay in this attempt even when its
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
			must(t, removeAllForce(root))
		}
	})
	project := copyPrivateTensorProject(t, root, "")
	script := filepath.Join(project, "recipe.py")
	body, err := os.ReadFile(script)
	must(t, err)
	seed := strings.Replace(string(body), "    await candidate(source=original, factor=0)", "    return original", 1)
	seed = strings.Replace(seed, "async def main():", "async def main() -> ModelArtifact:", 1)
	seed = strings.Replace(seed, "from tensor_source import", "from cozy_runtime.author import ModelArtifact\nfrom tensor_source import", 1)
	must(t, os.WriteFile(script, []byte(seed), 0600))
	if status, output := runCozyPath(t, root, path, "run", script, "--await", "--json"); status != 0 {
		t.Fatalf("native fixture seed failed [%d]: %s", status, output)
	}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	producer, problem := store.RequestByReference("1")
	fatal(t, problem)
	if producer == nil || producer.State != "succeeded" {
		t.Fatalf("native seed did not complete: %+v", producer)
	}
	weights, problem := store.MachineModelRetentions(producer.ID)
	fatal(t, problem)
	if len(weights) != 1 || weights[0].State != "held" {
		t.Fatalf("native seed has no verified recipient custody: %+v", weights)
	}
	seedModel, problem := records.DecodeModelArtifact(weights[0].Artifact)
	fatal(t, problem)
	// Register the already-produced bytes as an ordinary local checkpoint.
	register := exec.Command(filepath.Join(control, "bin", "tfs"), "local", "replace", filepath.Join(root, "tensorfs"), "library-seed", strings.TrimPrefix(seedModel.Manifest.Digest, "sha256:"), seedModel.Manifest.Digest, strconv.FormatInt(seedModel.Manifest.Length, 10), "--observed", "absent") //cozy:allow fixture uses native repository CLI to register a real produced checkpoint
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
import io
import struct
from cozy_runtime.author import Context, ModelArtifact
from tensorfs.derived import Derivation, Part, Target, Tensor
from tensor_candidate import Source, PLAIN, scale_values

def main(ctx: Context, *, source: Source) -> ModelArtifact:
    tensor = Tensor(logical_dtype="f32", shape=(512,), encoding=PLAIN,
                    parts={"value": Part("f32", (512,))})
    with ctx.output("weights").open(Derivation(sources={"original": ctx.tensorfs_source(source)},
        targets={"model": Target(source="original", source_component="model",
            drop=("weight",), add={"weight": tensor})},
        configs={}, order=(("model", "weight"),))) as writer:
        if writer.receipt is not None:
            return ctx.adopt_model(writer.receipt)
        data = bytearray(2048)
        writer.source_read_into("original", "model", "weight", "value", 0, data)
        writer.add_part("model", "weight", "value",
                        io.BytesIO(struct.pack("<512f", *scale_values(struct.unpack("<512f", data)))))
        return ctx.adopt_model(writer.commit())
`
	must(t, os.WriteFile(script, []byte(code), 0600))
	status, output := runCozyPath(t, root, path, "run", script, "model.source=local/library-seed#"+seedModel.Manifest.Digest, "--await", "--json")
	if status != 0 {
		t.Fatalf("ordinary library script failed [%d]: %s", status, output)
	}
	request, problem := store.RequestByReference("2")
	fatal(t, problem)
	if request == nil || request.State != "succeeded" || len(request.Models) != 1 || request.WeightsOutputs == "" {
		t.Fatalf("ordinary model/weights attempt missing: %+v", request)
	}
	children, problem := store.Children(request.ID)
	fatal(t, problem)
	if len(children) != 0 {
		t.Fatalf("helper import produced child requests: %+v", children)
	}
	if calls := machineChildren(t, root, store, "2"); len(calls) != 0 {
		t.Fatalf("ordinary helper import dispatched a managed child: %+v", calls)
	}
	outputs, problem := store.MachineModelRetentions(request.ID)
	fatal(t, problem)
	if len(outputs) != 1 || outputs[0].State != "held" {
		t.Fatalf("ordinary native output missing: %+v", outputs)
	}
	result, problem := records.DecodeModelArtifact(outputs[0].Artifact)
	fatal(t, problem)
	read := exec.Command(filepath.Join(control, "bin", "python"), filepath.Join("testdata", "private_child_read.py"), result.Manifest.Digest, "14", filepath.Join(root, "tensorfs"))
	if output, err := read.CombinedOutput(); err != nil {
		t.Fatalf("real transformed tensor: %v %s", err, output)
	}
	t.Log(fmt.Sprintf("one ordinary request read native Model and wrote value 14; %s", result.Manifest.Digest))
}
