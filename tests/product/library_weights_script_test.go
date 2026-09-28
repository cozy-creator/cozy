package producttest

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

// An ordinary script's native TensorFS output stays in its own attempt on this computer's
// machine even when a helper library it imports also exports a managed callable: the import
// dispatches nothing. No worker is mocked.
func TestPrivateLibraryScriptKeepsWeightsInItsOwnAttempt(t *testing.T) {
	integration(t)
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
from tensor_candidate import PLAIN, scale_values

def main(ctx: Context) -> ModelArtifact:
    tensor = Tensor(logical_dtype="f32", shape=(512,), encoding=PLAIN,
                    parts={"value": Part("f32", (512,))})
    with ctx.output("weights").open(Derivation(sources={},
        targets={"model": Target(add={"weight": tensor})},
        configs={}, order=(("model", "weight"),))) as writer:
        if writer.receipt is not None:
            return ctx.adopt_model(writer.receipt)
        writer.add_part("model", "weight", "value",
                        io.BytesIO(struct.pack("<512f", *scale_values([7.0] * 512))))
        return ctx.adopt_model(writer.commit())
`
	must(t, os.WriteFile(script, []byte(code), 0600))
	status, output := runCozyPath(t, root, path, "run", script, "--await", "--json")
	if status != 0 {
		t.Fatalf("ordinary library script failed [%d]: %s", status, output)
	}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	request, problem := store.RequestByReference("1")
	fatal(t, problem)
	if request == nil || request.State != "succeeded" || request.WeightsOutputs == "" {
		t.Fatalf("ordinary weights attempt missing: %+v", request)
	}
	children, problem := store.Children(request.ID)
	fatal(t, problem)
	if len(children) != 0 {
		t.Fatalf("helper import produced child requests: %+v", children)
	}
	if calls := machineChildren(t, root, store, "1"); len(calls) != 0 {
		t.Fatalf("ordinary helper import dispatched a managed child: %+v", calls)
	}
	outputs, problem := store.MachineModelRetentions(request.ID)
	fatal(t, problem)
	if len(outputs) != 1 || outputs[0].State != "held" {
		t.Fatalf("ordinary native output missing: %+v", outputs)
	}
	result, problem := records.DecodeModelArtifact(outputs[0].Artifact)
	fatal(t, problem)
	read := exec.Command(filepath.Join(control, "bin", "python"), filepath.Join("testdata", "private_child_read.py"), result.Manifest.Digest, "14", machineStore(root))
	if output, err := read.CombinedOutput(); err != nil {
		t.Fatalf("real transformed tensor: %v %s", err, output)
	}
	t.Logf("one ordinary request wrote value 14 through its helper; %s", result.Manifest.Digest)
}
