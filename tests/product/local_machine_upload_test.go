package producttest

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"
)

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
dependencies=["cozy-runtime>=`+runtimeFloor+`", "tensorfs>=0.3.74,<0.4"]
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

// A conversion shows every call it made: its own execution as call 0, and its checkpoint
// upload as a call, with how it ended and why. Here the Hub refuses the publication, so the
// run fails and its upload call says so, with the Hub's reason. The machine records the call
// for the publication it makes during the run.
func TestAConversionShowsItsCheckpointUploadAsACall(t *testing.T) {
	h, root, _, _ := parityMachines(t)
	// The machine's grant to publish into the destination the owner consented to, and the
	// short token its Runtime renews with its own worker capability.
	grant := func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"token": "machine-grant", "expires_at": time.Now().Add(time.Hour).UTC().Format(time.RFC3339)})
	}
	refuse := func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusConflict)
		_, _ = w.Write([]byte(`{"error":{"code":"publication.destination_refused","message":"the destination refuses this checkpoint"}}`))
	}
	h.mux.HandleFunc("POST /v1/machine-authorizations", grant)
	worker := h.worker.Config.Handler
	h.worker.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && strings.HasPrefix(r.URL.Path, "/v1/worker/machine-authorizations/"):
			grant(w, r)
		case strings.HasPrefix(r.URL.Path, "/v1/models/"):
			refuse(w, r)
		default:
			worker.ServeHTTP(w, r)
		}
	})
	if code, out := runCozy(t, root, "package", "install", weightsProject(t), "--editable"); code != 0 {
		t.Fatalf("installing the weights package [exit %d]\n%s", code, out)
	}
	code, out := runCozy(t, root, "run", "local/weights-proof/convert", "n=3", "--upload-to", "proof/model", "--await", "--json")
	if code == 0 || !strings.Contains(out, "publication was refused before its commit") {
		t.Fatalf("a refused upload did not fail its conversion with the reason [exit %d]\n%s", code, out)
	}
	code, out = runCozy(t, root, "run", "show", "1", "--json")
	var shown struct {
		Calls []struct {
			Number   int    `json:"number"`
			Function string `json:"function"`
			Label    string `json:"label"`
			Status   string `json:"status"`
			Error    string `json:"error"`
		} `json:"calls"`
	}
	if code != 0 || json.Unmarshal([]byte(out), &shown) != nil || len(shown.Calls) != 2 {
		t.Fatalf("run show --json does not list the run and its upload [exit %d]\n%.3000s", code, out)
	}
	run, upload := shown.Calls[0], shown.Calls[1]
	if run.Number != 0 || run.Function != "convert" || run.Status != "failed" {
		t.Fatalf("call 0 is not the run's own execution: %+v", run)
	}
	if upload.Number != 1 || upload.Function != "upload_checkpoint" || upload.Label != "Upload checkpoint to proof/model" ||
		upload.Status != "failed" || !strings.HasPrefix(upload.Error, "publication was refused before its commit: publication.destination_refused") {
		t.Fatalf("the checkpoint upload is not a call that says why it failed: %+v", upload)
	}
	code, human := runCozy(t, root, "run", "show", "1")
	t.Logf("cozy run show 1:\n%s", human)
	if code != 0 || !strings.Contains(human, "calls (1)") ||
		!regexp.MustCompile(`(?m)^1 +Upload checkpoint to proof/model +upload_checkpoint +failed `).MatchString(human) {
		t.Fatalf("run show does not list the upload:\n%s", human)
	}
}
