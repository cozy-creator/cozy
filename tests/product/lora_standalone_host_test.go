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
	if !*machineLoRAServingGPU {
		t.Setenv("CUDA_VISIBLE_DEVICES", "")
	}
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
	// Keep the arithmetic proof on CPU; the machine still uses the same public
	// Runtime, native TensorFS store, authenticated broker and preparation path.
	if !*machineLoRAServingGPU {
		virtualInventoryFor(t, host.Root(), true)
	} else {
		device, err := exec.Command("nvidia-smi", "--query-gpu=name", "--format=csv,noheader").Output()
		if err != nil || !strings.Contains(string(device), "H100") {
			t.Fatal("-lora-serving-gpu requires the isolated rented H100 host")
		}
	}
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
	connection, problem := resolver.DialAt(t.Context(), machines.Local, h.server.URL, "LoRA preparation proof", true)
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
dependencies=["cozy-runtime[adapters]>=%s","torch==2.13.0","transformers>=5.16.1,<6"]
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
from cozy_runtime.author import AdapterCompatibility, App, Config, Context, Loader, Model, invocable, uses_components
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
    @uses_components('transformer')
    def score(self)->int:
        component=self.pipe.components['transformer']
        x=torch.tensor([[1,2,3]],dtype=torch.float32,device=next(component.parameters()).device)
        return int(4*component.proj(x).sum().item())
class Request(msgspec.Struct): pass
class Result(msgspec.Struct): value:int
app=App()
@app.job
def scalar(payload:Request)->Result: return Result(42)
@invocable(defaults={'model':[{'gpu':'*','lane':'proof/base@1.0.0/fp32'}]})
async def generate(ctx:Context,*,payload:Request,model:Probe)->Result: return Result(model.score())
app.entrypoint(generate)
@app.job
async def child(ctx:Context,payload:Request)->Result:
    return await generate(payload=payload)
`), 0600))
	lock := exec.Command("uv", "lock", "--project", project, "--python", "3.12")
	lock.Env = append(childEnv(t, root), "UV_TORCH_BACKEND=cpu")
	if out, err := lock.CombinedOutput(); err != nil {
		t.Fatalf("lock private probe: %v\n%s", err, out)
	}
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
	// A normal local root prepares its selected stack before acceptance. First land a
	// baseline, so an adapter run must not reuse the already-held base placement.
	if *machineLoRAServingGPU {
		views := map[string]string{}
		executors := map[string]string{}
		for _, arm := range []struct {
			key, function, slot string
			scales              []string
			score               int
		}{
			{"baseline", "generate", "model", nil, 184},
			{"root-stack", "generate", "model", []string{"0.5", "-0.25"}, 320},
			{"root-repeat", "generate", "model", []string{"0.5", "-0.25"}, 320},
			{"root-reversed", "generate", "model", []string{"-0.25", "0.5"}, 320},
			{"root-zero", "generate", "model", []string{"0", "-0.25"}, 124},
			{"root-baseline-after", "generate", "model", nil, 184},
			{"child-stack", "child", "generate.models.model", []string{"0.5", "-0.25"}, 320},
		} {
			args := []string{"run", "local/lora-host-probe/" + arm.function, "--await", "--json", "--idempotency-key", arm.key}
			if arm.function == "generate" {
				args = append(args, "payload:={}", "model.model=proof/base@1.0.0/fp32")
			}
			for i, scale := range arm.scales {
				name := []string{"first", "second"}[i]
				if arm.key == "root-reversed" {
					name = []string{"second", "first"}[i]
				}
				args = append(args, "--lora", arm.slot+":transformer=proof/"+name+"@1.0.0/fp32,"+scale)
			}
			if code, out := runCozy(t, root, args...); code != 0 || !strings.Contains(out, fmt.Sprintf(`"value":%d`, arm.score)) {
				t.Fatalf("%s CLI serving [%d], expected score %d: %s", arm.key, code, arm.score, out)
			}
			request, problem := store.RequestByIdempotencyKey(arm.key)
			fatal(t, problem)
			if request == nil {
				t.Fatal("normal CLI lost its request")
			}
			if arm.function == "generate" {
				events, problem := store.EvidenceEvents(request.ID, 1000)
				fatal(t, problem)
				for _, event := range events {
					if event.Type == "machine.executor" {
						executors[arm.key] = fmt.Sprintf("%v", event.Payload["pid"])
					}
				}
				if executors[arm.key] == "" || executors[arm.key] == "<nil>" || executors[arm.key] == "0" {
					t.Fatalf("%s retained no real executor identity", arm.key)
				}
				if arm.key == "root-repeat" && executors[arm.key] != executors["root-stack"] {
					t.Fatal("identical ordered stack replaced its warm executor")
				}
			}
			// Read the actual Runtime journal. The invocation's capture alone does not
			// prove which placement was admitted and later handed to its executor.
			read := exec.Command(python, "-I", "-c", `import json,sqlite3,sys
from pathlib import Path
root=Path(sys.argv[1]); db=sqlite3.connect('file:'+str(root/'.cozy-workspace/journal.sqlite3')+'?mode=ro',uri=True)
row=db.execute('select preparation from executions where request=?',(sys.argv[2],)).fetchone()
if sys.argv[3]=='child':
    call=db.execute('select e.preparation from executions e join execution_calls c on e.request=c.child_request where c.parent_request=?',(sys.argv[2],)).fetchone()
    if call is not None: row=call
prepared=json.loads(row[0]); slots=[]
for installation in prepared['installations'].values():
    placement=installation['placement']; models={model['id']:model for model in placement.get('models',[])}
    for entry in placement.get('entrypoints',[]):
        if entry['name']!='generate': continue
        for slot in entry.get('slots',[]):
            stack=[dict(adapter,manifest=models[adapter['model_id']]['manifest']['digest']) for adapter in slot.get('adapters',[])]
            components=[models[component['model_id']]['manifest']['digest'] for component in slot.get('components',[])]
            slots.append({'stack':stack,'base':models[slot['reference_model_id']]['manifest']['digest'],'components':components})
print(json.dumps(slots))
`, filepath.Join(root, "tensorfs"), request.ID, arm.function)
			raw, err := read.CombinedOutput()
			if err != nil {
				t.Fatalf("%s retained preparation: %v\n%s", arm.key, err, raw)
			}
			var slots []struct {
				Stack      []struct{ Scale, Manifest string } `json:"stack"`
				Base       string                             `json:"base"`
				Components []string                           `json:"components"`
			}
			must(t, json.Unmarshal(raw, &slots))
			if len(slots) != 1 {
				t.Fatalf("%s selected no exact serving slot: %s", arm.key, raw)
			}
			if slots[0].Base != seeded["base"].Record["manifest_id"] {
				t.Fatalf("%s replaced the original base custody", arm.key)
			}
			if len(slots[0].Components) != 1 || (slots[0].Components[0] != slots[0].Base) != (len(arm.scales) > 0) {
				t.Fatalf("%s served no selected component view: %+v", arm.key, slots[0])
			}
			views[arm.key] = slots[0].Components[0]
			if arm.key == "root-repeat" || arm.key == "child-stack" {
				if views[arm.key] != views["root-stack"] {
					t.Fatalf("%s changed the same ordered component view", arm.key)
				}
			}
			if arm.key == "root-reversed" || arm.key == "root-zero" {
				if views[arm.key] == views["root-stack"] {
					t.Fatalf("%s reused a different stack's component view", arm.key)
				}
			}
			stack := slots[0].Stack
			if len(stack) != len(arm.scales) {
				t.Fatalf("%s served %d adapters; requested %d", arm.key, len(stack), len(arm.scales))
			}
			t.Logf("%s: score=%d adapters=%d component=%s", arm.key, arm.score, len(stack), views[arm.key])
			for i, adapter := range stack {
				name := []string{"first", "second"}[i]
				if arm.key == "root-reversed" {
					name = []string{"second", "first"}[i]
				}
				if adapter.Manifest != seeded[name].Record["manifest_id"] {
					t.Fatalf("%s changed adapter %d selection", arm.key, i)
				}
				if adapter.Scale != arm.scales[i] {
					t.Fatalf("%s changed adapter %d strength: %s", arm.key, i, adapter.Scale)
				}
			}
		}
	}
	// The daemon now owns Control; this independent reader reuses the same owner.
	resolver.Held = func(string) bool { return true }
	resolver.Forget(machines.Local)
	connection, problem = resolver.DialAt(t.Context(), machines.Local, h.server.URL, "LoRA preparation readback", true)
	fatal(t, problem)
	transfer, problem := store.MachinePackageTransfer(row.ID, connection.Claim.WorkerBootId, row.LocalInstallationID)
	fatal(t, problem)
	if transfer.Operation == "" {
		t.Fatal("normal CLI did not retain its source preparation operation")
	}
	const token = "fake-private-preparation-token-46a96b1e"
	selected := &pb.DesiredPrivatePlacementSet{OperationId: transfer.Operation, InstallationId: row.LocalInstallationID,
		Hub: connection.Hub, Owner: "fixture-account", SourceCredentials: []*pb.SourceCredential{{
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
	must(t, canonical.Unmarshal(prepared.PlacementSetCanonicalBytes, &placement))
	if len(placement.Placements) != 1 {
		t.Fatal("private preparation returned additional installations")
	}
	if placement.Placements[0].InstallationId != row.LocalInstallationID {
		t.Fatal("private preparation replaced its accepted root installation")
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
	scan := func() {
		t.Helper()
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
	// Inspect live WAL files before shutdown can checkpoint or remove them.
	scan()
	_, _ = runCozy(t, root, "down")
	fatal(t, host.Stop(t.Context()))
	scan()
}
