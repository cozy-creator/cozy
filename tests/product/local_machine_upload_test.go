package producttest

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/machines"
)

// `cozy run upload` from this computer's machine: a job there retains its weights output,
// and the upload sends it from that machine to the Hub over the machine connection, as it
// does from a rental; this computer's machine is never taken for an ended rental.
func TestAnOutputRetainedOnThisComputersMachineUploads(t *testing.T) {
	h, root, _, store := parityMachines(t)
	checkpoints := serveCheckpoints(t, h.fakeRentalHub)
	if code, out := runCozy(t, root, "package", "install", weightsProject(t), "--editable"); code != 0 {
		t.Fatalf("installing the weights package [exit %d]\n%s", code, out)
	}
	code, out := runCozy(t, root, "run", "local/weights-proof/convert", "n=2", "--await", "--json", "--idempotency-key", "retained")
	if code != 0 {
		t.Fatalf("the weights job failed [exit %d]\n%s", code, out)
	}
	row, problem := store.RequestByIdempotencyKey("retained")
	fatal(t, problem)
	link, problem := store.MachineExecution(row.ID)
	fatal(t, problem)
	if link == nil || link.MachineID != machines.Local {
		t.Fatalf("the weights job did not run on this computer's machine: %+v", link)
	}
	code, out = runCozy(t, root, "run", "upload", row.ID+"#model", "proof/model", "--await", "--json")
	var uploaded struct {
		State      string `json:"state"`
		Checkpoint string `json:"checkpoint"`
	}
	if code != 0 || json.Unmarshal([]byte(out), &uploaded) != nil || uploaded.State != "uploaded" {
		t.Fatalf("the retained output was not uploaded from this computer's machine [exit %d]\n%s", code, out)
	}
	checkpoints.mu.Lock()
	defer checkpoints.mu.Unlock()
	if checkpoints.finalized[uploaded.Checkpoint] == "" || len(checkpoints.stored) == 0 {
		t.Fatalf("the Hub holds no finalized checkpoint %s: %v", uploaded.Checkpoint, checkpoints.finalized)
	}
}

// weightsProject is local/weights-proof: one job that derives a small checkpoint into its
// declared weights output.
func weightsProject(t *testing.T) string {
	t.Helper()
	project := filepath.Join(t.TempDir(), "weights-proof")
	must(t, os.MkdirAll(project, 0o700))
	sources := ""
	if *machineRuntimeWheel != "" {
		sources = fmt.Sprintf("[tool.uv.sources]\ncozy-runtime={path=%q}\ntensorfs={path=%q}\n", *machineRuntimeWheel, *machineTensorFSWheel)
	}
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(`[project]
name="weights-proof"
version="0.0.1"
requires-python=">=3.12,<3.13"
dependencies=["cozy-runtime>=`+machines.RuntimeFloor+`", "tensorfs>=0.3.74,<0.4"]
[project.entry-points."cozy.application"]
default="weights_proof:app"
`+sources+`[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
only-include=["weights_proof.py"]
`), 0o600))
	must(t, os.WriteFile(filepath.Join(project, "package.toml"), []byte("[application]\nobject=\"weights_proof:app\"\n"), 0o600))
	must(t, os.WriteFile(filepath.Join(project, "weights_proof.py"), []byte(`import struct

import tensorfs
from tensorfs.derived import Config, Derivation, Part, Target, Tensor

from cozy_runtime.author import App, Context, ModelArtifact, WeightsOutput, invocable

app = App()
PLAIN = dict(tensorfs.seed_digests())["plain/1"]


@invocable(memoize=True)
async def convert(ctx: Context, *, n: int) -> ModelArtifact:
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
            output.add_part("body", "layer.weight", "value", struct.pack("<512e", *(i / n for i in range(512))))
            output.add_config("pipeline", b"{}")
        receipt = output.receipt or output.commit()
    return ctx.adopt_model(receipt)


app.job(convert, weights=(WeightsOutput("model", max_new_bytes=65536),))
`), 0o600))
	if out, err := exec.Command("uv", "lock", "--project", project).CombinedOutput(); err != nil {
		t.Fatalf("locking the weights package: %v\n%s", err, out)
	}
	return project
}
