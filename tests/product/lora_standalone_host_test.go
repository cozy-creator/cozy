package producttest

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/secret"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Install private code through the normal CLI, then prepare its unused serving
// slot through the real authenticated standalone agent. No GPU execution is
// needed to inspect the ordered, derived adapter view returned by Runtime.
func TestStandaloneHostPreparesOrderedPrivateLoRAView(t *testing.T) {
	integration(t)
	if *machineHostBinary == "" || *machineRuntimeWheel == "" || *machineTensorFSWheel == "" {
		t.Skip("requires an exact standalone agent and paired Runtime/TensorFS wheels")
	}
	h := newMachineHub(t)
	root, err := os.MkdirTemp("", "cz-lora-")
	must(t, err)
	claimScratch(root)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+h.server.URL+
		"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0600))
	provisionMachine(t, root)
	layout, problem := home.Open(root)
	fatal(t, problem)
	host := machines.NewHost(layout.Machine, filepath.Join(root, "tensorfs"), nil)
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "down")
		_ = host.Stop(context.Background())
		if t.Failed() {
			t.Log("standalone LoRA evidence retained at", root)
		} else {
			_ = removeAllForce(root)
		}
	})
	resolver := &machines.Resolver{Host: host, HubOrigin: h.server.URL,
		Hub: func(origin string) *hub.Client {
			return hub.New(config.Config{HubURL: origin, HubToken: secret.New("rental-idle-test")}, "LoRA preparation proof")
		}}
	connection, problem := resolver.Dial(t.Context(), machines.Local, "LoRA preparation proof")
	fatal(t, problem)
	defer resolver.Forget(machines.Local)
	var workspace *pb.MachineExecutionWorkspace
	waitFor(t, root, "the standalone workspace to finish startup", func() bool {
		var err error
		workspace, err = connection.Host.GetMachineExecutionWorkspace(t.Context(), &pb.MachineExecutionWorkspaceQuery{Claim: connection.Claim})
		if status.Code(err) == codes.Unavailable {
			return false
		}
		must(t, err)
		return workspace != nil
	})
	if !workspace.ModelOverrides || connection.WireMinor < 70 {
		if *requireMachineHost {
			t.Fatal("the required standalone pair does not support generic model overrides")
		}
		t.Skip("the selected published pair predates generic model overrides")
	}
	data, err := os.ReadFile("testdata/lora-cpu-checkpoints.json")
	must(t, err)
	var seeded map[string]struct {
		Record map[string]any `json:"record"`
	}
	must(t, json.Unmarshal(data, &seeded))
	seedFile := filepath.Join(root, "seed-data.json")
	must(t, os.WriteFile(seedFile, data, 0600))
	python := filepath.Join(host.Root(), "opt/cozy/python/bin/python")
	seed := exec.Command(python, "-I", "-c", `import base64, json, os, sys, tempfile
import tensorfs
rows=json.load(open(sys.argv[2])); store=tensorfs.Store.open(sys.argv[1])
with tempfile.TemporaryDirectory() as scratch:
    for name,row in rows.items():
        manifest=base64.b64decode(row['manifest']); entries=json.loads(manifest)['entries']
        for entry in entries:
            blob=entry['blob']; raw=base64.b64decode(row['files'][entry['path']])
            path=os.path.join(scratch,name+'-'+entry['path']); open(path,'wb').write(raw)
            store.put_file(path,'sha256:'+blob['sha256'],blob['length'])
        record=row['record']; store.put_manifest(manifest,record['manifest_id'],record['manifest_length'])
        operation=store.begin_operation('seed-'+name,'proof',name)
        operation.hold_manifest(record['manifest_id'],record['manifest_length'])
        operation.commit_release(None,'1.0.0','fp32',record['manifest_id'],record['manifest_length'])
`, filepath.Join(root, "tensorfs"), seedFile)
	if out, err := seed.CombinedOutput(); err != nil {
		t.Fatalf("seed native CPU checkpoints: %v\n%s", err, out)
	}
	h.mux.HandleFunc("GET /v1/models/resolve", func(w http.ResponseWriter, r *http.Request) {
		for name, row := range seeded {
			if r.URL.Query().Get("ref") == "proof/"+name+"@1.0.0" {
				_ = json.NewEncoder(w).Encode(row.Record)
				return
			}
		}
		http.NotFound(w, r)
	})
	h.mux.HandleFunc("GET /v1/models/proof/{name}", func(w http.ResponseWriter, r *http.Request) {
		if _, exists := seeded[r.PathValue("name")]; !exists {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write([]byte(`{"releases":[{"release":"1.0.0","lanes":[{"lane":"fp32","bytes":1000}]}]}`))
	})
	doors := h.worker.Config.Handler
	h.worker.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models/resolve" || strings.HasPrefix(r.URL.Path, "/v1/models/proof/") {
			h.mux.ServeHTTP(w, r)
			return
		}
		doors.ServeHTTP(w, r)
	})
	project := filepath.Join(root, "project")
	must(t, os.MkdirAll(project, 0700))
	metadata := fmt.Sprintf(`[project]
name="lora-host-probe"
version="0.1.0"
requires-python=">=3.12,<3.13"
dependencies=["cozy-runtime[adapters]>=%s","torch==2.13.0"]
[tool.uv.sources]
cozy-runtime={path=%q}
tensorfs={path=%q}
[project.entry-points."cozy.application"]
default="lora_host_probe:app"
[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
only-include=["lora_host_probe.py"]
`, runtimeFixtureVersion(t, *machineRuntimeWheel), *machineRuntimeWheel, *machineTensorFSWheel)
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(metadata), 0600))
	must(t, os.WriteFile(filepath.Join(project, "package.toml"), []byte("[application]\nobject='lora_host_probe:app'\n"), 0600))
	must(t, os.WriteFile(filepath.Join(project, "lora_host_probe.py"), []byte(`import msgspec
import torch
from cozy_runtime.author import AdapterCompatibility, App, Config, Loader, Model
class Pipeline:
    def __init__(self):
        transformer=torch.nn.Module()
        transformer.proj=torch.nn.Linear(3,2)
        transformer.untouched=torch.nn.Linear(1,1,bias=False)
        self.components={'transformer':transformer}
def build(config: Config)->Pipeline: return Pipeline()
class Probe(Model[Pipeline]):
    __adapter_compatibility__=(AdapterCompatibility('lora','',('transformer',)),)
    def load(self,loader:Loader)->None: self.pipe=loader.construct(Pipeline,factory=build)
class Request(msgspec.Struct): pass
class Result(msgspec.Struct): value:int
app=App()
@app.job
def scalar(payload:Request)->Result: return Result(42)
@app.entrypoint
def generate(payload:Request,model:Probe)->Result: return Result(0)
`), 0600))
	if code, out := runCozy(t, root, "package", "install", project, "--editable"); code != 0 {
		t.Fatalf("install private probe [%d]: %s", code, out)
	}
	if code, out := runCozy(t, root, "run", "local/lora-host-probe/scalar", "--await", "--json", "--idempotency-key", "land-private-code"); code != 0 {
		t.Fatalf("land private code [%d]: %s", code, out)
	}
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	row, problem := store.RequestByIdempotencyKey("land-private-code")
	fatal(t, problem)
	if row == nil || row.LocalInstallationID == "" {
		t.Fatal("normal CLI did not retain its private installation identity")
	}
	// The daemon now owns Control; this independent reader reuses the same owner.
	resolver.Held = func(string) bool { return true }
	resolver.Forget(machines.Local)
	connection, problem = resolver.Dial(t.Context(), machines.Local, "LoRA preparation readback")
	fatal(t, problem)
	const token = "fake-private-preparation-token-46a96b1e"
	selected := &pb.DesiredPrivatePlacementSet{OperationId: "lora-private-proof", InstallationId: row.LocalInstallationID,
		Hub: h.server.URL, Owner: "fixture-account", SourceCredentials: []*pb.SourceCredential{{
			Provider: pb.NativeSourceOperation_NATIVE_SOURCE_OPERATION_HUGGINGFACE, Credential: "bearer " + token}},
		ModelChoices: []*pb.ModelChoice{{Parameter: "generate.models.model", Repository: "proof/base", Release: "1.0.0", Lane: "fp32",
			Adapters: []*pb.DownloadAdapterRef{{Model: "proof/first", Release: "1.0.0", Lane: "fp32", Component: "transformer", SourceComponent: "adapter", Scale: "0.5"},
				{Model: "proof/second", Release: "1.0.0", Lane: "fp32", Component: "transformer", SourceComponent: "adapter", Scale: "-0.25"}}}}}
	stream, err := connection.Host.PreparePrivatePlacement(t.Context(), &pb.PreparePrivatePlacementCall{Claim: connection.Claim, PrivatePlacementSet: selected})
	must(t, err)
	var prepared *pb.DesiredPlacementSet
	for {
		event, err := stream.Recv()
		if err == io.EOF {
			break
		}
		must(t, err)
		if event.Stage == pb.PrepareStage_PREPARE_STAGE_REFUSED {
			t.Fatalf("private LoRA preparation refused: %s: %s", event.SafeCode, event.SafeDetail)
		}
		if event.PlacementSet != nil {
			prepared = event.PlacementSet
		}
	}
	if prepared == nil {
		t.Fatal("Runtime returned no prepared adapter view")
	}
	var placement pb.PlacementSet
	_, err = canonical.Read(prepared.PlacementSetCanonicalBytes, &placement)
	must(t, err)
	if len(placement.Placements) != 1 {
		t.Fatal("private preparation returned additional installations")
	}
	var slot *pb.Slot
	for _, entry := range placement.Placements[0].Entrypoints {
		if entry.Name == "generate" && len(entry.Slots) == 1 {
			slot = entry.Slots[0]
		}
	}
	if slot == nil || len(slot.Adapters) != 2 || slot.Adapters[0].Scale != "0.5" || slot.Adapters[1].Scale != "-0.25" {
		t.Fatal("Runtime omitted or reordered the actual prepared adapter stack")
	}
	var original, composed *pb.Model
	for _, model := range placement.Placements[0].Models {
		if model.Id == slot.ReferenceModelId {
			original = model
		}
		for _, component := range slot.Components {
			if component.Component == "transformer" && component.ModelId == model.Id {
				composed = model
			}
		}
	}
	if original == nil || composed == nil || original.Manifest == nil || composed.Manifest == nil || bytes.Equal(original.Manifest.Digest, composed.Manifest.Digest) {
		t.Fatal("Runtime returned no independently derived component view")
	}
	if original.Repo != "proof/base" {
		t.Fatal("private preparation replaced original base custody")
	}
	baseID, err := canonical.Spell(original.Manifest.Digest)
	must(t, err)
	if baseID != seeded["base"].Record["manifest_id"] {
		t.Fatal("private preparation changed the original base checkpoint")
	}
	if bytes.Contains(prepared.PlacementSetCanonicalBytes, []byte(token)) {
		t.Fatal("private preparation retained its source credential")
	}
	_, _ = runCozy(t, root, "down")
	fatal(t, host.Stop(t.Context()))
	for _, directory := range []string{root, host.Root()} {
		must(t, filepath.Walk(directory, func(path string, info os.FileInfo, err error) error {
			if err != nil || info.IsDir() || info.Mode()&os.ModeType != 0 {
				return err
			}
			body, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			if bytes.Contains(body, []byte(token)) {
				t.Errorf("private preparation credential persisted in %s", path)
			}
			return nil
		}))
	}
}
