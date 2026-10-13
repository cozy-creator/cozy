package producttest

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// A published job run on this computer with --upload-to takes its Tree input like every other
// job: Calcifer captures the directory and the machine materializes it for the callable,
// whether it is bound with --input-tree or --asset. The stand-in Hub serves no checkpoint
// publication, so each run ends at its upload call, after the job ran against the tree.
func TestAPublishedUploadJobStagesItsInputTree(t *testing.T) {
	h, root, _, store := parityMachines(t)
	h.hubAccess.signIn(t, root)
	project := parityProject(t)
	pyproject := filepath.Join(project, "pyproject.toml")
	raw, err := os.ReadFile(pyproject)
	must(t, err)
	must(t, os.WriteFile(pyproject, []byte(strings.Replace(string(raw), `"msgspec>=0.19"`, `"msgspec>=0.19", "tensorfs>=0.3.74"`, 1)), 0o600))
	must(t, os.WriteFile(filepath.Join(project, "machine_parity.py"), []byte(`import struct

import tensorfs
from tensorfs.derived import Config, Derivation, Part, Target, Tensor

from cozy_runtime.author import App, Context, ModelArtifact, Tree, WeightsOutput, invocable

app = App()
PLAIN = dict(tensorfs.seed_digests())["plain/1"]


@invocable()
async def convert(ctx: Context, *, files: Tree) -> ModelArtifact:
    names = sorted(str(path.relative_to(files.path)) for path in files.files())
    if names != ["a.txt", "nested/b.bin"]:
        raise ValueError(f"the tree arrived as {names}")
    definition = Derivation(
        sources={},
        targets={"body": Target(add={
            "layer.weight": Tensor("f16", (16, 32), PLAIN, {"value": Part("f16", (16, 32))}),
        })},
        configs={"pipeline": Config("add")},
        order=(("body", "layer.weight"),),
    )
    with tensorfs.derive(ctx.output("model"), definition) as output:
        if output.receipt is None:
            output.add_part("body", "layer.weight", "value", struct.pack("<512e", *(i / len(names) for i in range(512))))
            output.add_config("pipeline", b"{}")
        receipt = output.receipt or output.commit()
    return ctx.adopt_model(receipt)


app.job(convert, weights=(WeightsOutput("model", max_new_bytes=65536),))
`), 0o600))
	if out, err := exec.Command("uv", "lock", "--project", project).CombinedOutput(); err != nil {
		t.Fatalf("locking the tree job: %v\n%s", err, out)
	}
	publishParityRelease(t, h, root, project)
	if code, out := runCozy(t, root, "package", "install", parityPublished); code != 0 {
		t.Fatalf("install here [exit %d]\n%s", code, out)
	}
	data := filepath.Join(root, "data")
	must(t, os.MkdirAll(filepath.Join(data, "nested"), 0o755))
	must(t, os.WriteFile(filepath.Join(data, "a.txt"), []byte("alpha"), 0o600))
	must(t, os.WriteFile(filepath.Join(data, "nested", "b.bin"), []byte("\x00\x01beta"), 0o600))
	for number, binding := range [][]string{{"files=data", "--input-tree", "data=" + data}, {"--asset", "files=" + data}} {
		key := "tree-upload-" + strconv.Itoa(number)
		args := append([]string{"run", parityPublished + "/convert", "--upload-to", "proof/model", "--await", "--json", "--idempotency-key", key}, binding...)
		code, out := runCozy(t, root, args...)
		if code == 0 || strings.Contains(out, "inputs_not_staged") || strings.Contains(out, "the tree arrived as") {
			t.Fatalf("%v did not run against its tree [exit %d]\n%s", binding, code, out)
		}
		request, problem := store.RequestByIdempotencyKey(key)
		fatal(t, problem)
		code, out = runCozy(t, root, "run", "show", request.ID, "--json")
		var shown struct {
			Calls []struct {
				Function, Status string
			} `json:"calls"`
		}
		if code != 0 || json.Unmarshal([]byte(out), &shown) != nil || len(shown.Calls) != 2 ||
			shown.Calls[0].Function != "convert" || shown.Calls[1].Function != "upload_checkpoint" || shown.Calls[1].Status != "failed" {
			t.Fatalf("%v did not reach its upload after running [exit %d]\n%.3000s", binding, code, out)
		}
	}
	// A rental the fleet places takes the tree too: it is captured before any machine is chosen.
	if code, out := runCozy(t, root, "run", parityPublished+"/convert", "--asset", "files="+data, "--upload-to", "proof/model",
		"--rental-only", "--json", "--idempotency-key", "tree-fleet"); code != 0 {
		t.Fatalf("a fleet-placed tree job was refused [exit %d]\n%s", code, out)
	}
	request, problem := store.RequestByIdempotencyKey("tree-fleet")
	fatal(t, problem)
	if request == nil || len(request.Assets) != 1 || request.Assets[0].Snapshot == nil || request.Assets[0].Snapshot.Reference == "" {
		t.Fatalf("the fleet-placed run did not capture its tree: %+v", request)
	}
}
