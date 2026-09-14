package producttest

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"io/fs"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

var localServingFixtureDir = flag.String("serving-fixture-dir", "", "Fixture source directory for local serving preparation qualification")

type servingSeed struct {
	Manifest         string           `json:"manifest"`
	Length           int64            `json:"length"`
	ManifestJSON     string           `json:"manifest_json"`
	HeaderDigest     string           `json:"header_digest"`
	SourceRoot       string           `json:"source_root"`
	HeaderComponents []string         `json:"header_components"`
	ComponentBytes   map[string]int64 `json:"component_bytes"`
	Bytes            int64            `json:"weight_bytes"`
}

func TestCapturedDevelopmentPlacementValidatesItsEnvironment(t *testing.T) {
	fixture := *localServingFixtureDir
	if fixture == "" {
		fixture = filepath.Join("testdata", "local_serving_preparation")
	}
	// These are actual PreparePrivatePlacement bytes from the native model proof,
	// produced by Runtime623dbb8; only the negative cases alter them.
	raw, err := os.ReadFile(filepath.Join(fixture, "placement-from-runtime-623dbb8.json"))
	must(t, err)
	check := func(raw []byte) *exit.Error {
		digest, err := canonical.Spell(canonical.Digest(raw))
		must(t, err)
		_, problem := orchestrator.PlacementFromExact("local/cozy-serving-preparation-fixture", "proof", digest, raw, nil)
		return problem
	}
	fatal(t, check(raw))
	for _, field := range []string{"local_revision_digest", "project_wheel", "environment_digest", "changed_environment"} {
		t.Run(field, func(t *testing.T) {
			doc, err := canonical.Read(raw, &pb.PlacementSet{})
			must(t, err)
			row := doc.List("placements")[0]
			switch field {
			case "environment_digest":
				delete(row, field)
			case "changed_environment":
				row["environment_digest"] = "sha256:" + strings.Repeat("f", 64)
			default:
				delete(row.Sub("development"), field)
			}
			changed, err := canonical.Write(doc)
			must(t, err)
			if check(changed) == nil {
				t.Fatalf("accepted captured placement with invalid %s", field)
			}
		})
	}
}

type servingPreparationEvent struct {
	Method                string `json:"method"`
	WorkerPID             int    `json:"worker_pid"`
	Birth                 string `json:"birth"`
	WorkerBoot            string `json:"worker_boot_id"`
	OperationID           string `json:"operation_id"`
	Placement             string `json:"placement"`
	Digest                string `json:"digest"`
	Error                 string `json:"error"`
	Environment           string `json:"environment"`
	DependencyEnvironment string `json:"dependency_environment"`
	Receipt               string `json:"receipt"`
}

// The catalog declares the exact real checkpoint seeded below. It does not return
// a placement, a successful preparation, a device claim, or generated output.
func servingModelCatalog(t *testing.T, seed servingSeed) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	var server *httptest.Server
	readObject := func(id string) ([]byte, error) {
		path := filepath.Join(t.TempDir(), "object")
		if err := exec.Command("tfs", "get", seed.SourceRoot, strings.TrimPrefix(id, "sha256:"), "--out", path).Run(); err != nil {
			return nil, err
		}
		return os.ReadFile(path)
	}
	mux.HandleFunc("GET /v1/models/proof/ordered", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(hub.ModelCard{
			Model: hub.Resource{Org: "proof", Name: "ordered"},
			Releases: []hub.ModelReleaseSummary{{
				ReleaseSummary: hub.ReleaseSummary{Release: "1.0.0"},
				Lanes: []hub.ModelLaneSummary{{Lane: "bf16", ManifestID: seed.Manifest,
					Components: seed.HeaderComponents, ComponentBytes: seed.ComponentBytes, Bytes: seed.Bytes}},
			}},
		})
	})
	mux.HandleFunc("GET /v1/models/proof/ordered/releases/1.0.0/lanes/bf16/manifest", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(seed.ManifestJSON))
	})
	mux.HandleFunc("GET /v1/models/resolve", func(w http.ResponseWriter, r *http.Request) {
		ref, lane := r.URL.Query().Get("ref"), r.URL.Query().Get("lane")
		if ref != "proof/ordered@1.0.0@"+seed.Manifest || lane != "bf16" {
			http.Error(w, "unknown fixture checkpoint", http.StatusNotFound)
			return
		}
		_ = json.NewEncoder(w).Encode(hub.ModelResolution{
			Model: "proof/ordered", Release: "1.0.0", Lane: "bf16",
			ManifestID: seed.Manifest, ManifestLength: seed.Length, HeaderID: seed.HeaderDigest,
			Bytes: seed.Bytes, Components: seed.HeaderComponents, ComponentBytes: seed.ComponentBytes,
		})
	})
	mux.HandleFunc("POST /v1/models/proof/ordered/releases/1.0.0/lanes/bf16/reads", func(w http.ResponseWriter, r *http.Request) {
		var ask struct {
			IDs []string `json:"object_ids"`
		}
		if err := json.NewDecoder(r.Body).Decode(&ask); err != nil {
			http.Error(w, "bad request", 400)
			return
		}
		reads := make([]hub.Read, 0, len(ask.IDs))
		for _, id := range ask.IDs {
			body, err := readObject(id)
			if err != nil {
				http.Error(w, "native object absent", 404)
				return
			}
			reads = append(reads, hub.Read{ObjectID: id, Length: int64(len(body)), URL: server.URL + "/objects/" + id})
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"reads": reads})
	})
	mux.HandleFunc("GET /objects/{id}", func(w http.ResponseWriter, r *http.Request) {
		body, err := readObject(r.PathValue("id"))
		if err != nil {
			http.Error(w, "native object absent", 404)
			return
		}
		_, _ = w.Write(body)
	})
	server = httptest.NewServer(mux)
	return server
}

func preparationEvents(t *testing.T, path string) []servingPreparationEvent {
	t.Helper()
	raw, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return nil
	}
	must(t, err)
	var result []servingPreparationEvent
	for _, line := range bytes.Split(bytes.TrimSpace(raw), []byte("\n")) {
		if len(line) == 0 {
			continue
		}
		var event servingPreparationEvent
		must(t, json.Unmarshal(line, &event))
		result = append(result, event)
	}
	return result
}

// This follows the same ordinary user commands as a local editable model package.
// Model initialization is observed in the real worker preparation RPC; the test
// never supplies a constructed component list to the owner or Runtime.
func TestLocalServingPreparationOwnsInitializationAndComponentOrder(t *testing.T) {
	if *privateChildRuntimeWheel == "" {
		t.Skip("requires the exact cl178 Runtime candidate wheel")
	}
	wheel, err := filepath.Abs(*privateChildRuntimeWheel)
	must(t, err)
	version := strings.Split(filepath.Base(wheel), "-")[1]
	fixture := *localServingFixtureDir
	if fixture == "" {
		fixture = filepath.Join("testdata", "local_serving_preparation")
	}
	for _, canary := range []bool{true, false} {
		t.Run(map[bool]string{true: "initialization_refusal", false: "prepared_components"}[canary], func(t *testing.T) {
			root, err := os.MkdirTemp("", "cozy-serving-preparation-")
			must(t, err)
			control := filepath.Join(t.TempDir(), "control")
			uv := func(args ...string) {
				t.Helper()
				out, err := exec.Command("uv", args...).CombinedOutput()
				if err != nil {
					t.Fatalf("uv %v: %v\n%s", args, err, out)
				}
			}
			uv("venv", "--python", "3.12", control)
			python := filepath.Join(control, "bin", "python")
			uv("pip", "install", "--python", python, wheel)
			seedCommand := exec.Command(python, filepath.Join(fixture, "seed.py"), filepath.Join(root, "tensorfs"))
			seedBytes, err := seedCommand.CombinedOutput()
			if err != nil {
				t.Fatalf("seed native checkpoint: %v\n%s", err, seedBytes)
			}
			var seed servingSeed
			must(t, json.Unmarshal(seedBytes, &seed))
			if !slices.Equal(seed.HeaderComponents, []string{"alpha", "spare", "zeta"}) || seed.Bytes != 48 {
				t.Fatalf("unexpected native fixture: %+v", seed)
			}
			catalog := servingModelCatalog(t, seed)
			defer catalog.Close()
			must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(
				"tensorhub_url: "+catalog.URL+"\ntensorhub_token: local-serving-fixture\ndaemon:\n  idle_shutdown_s: 0\n"), 0600))
			launcher, err := os.ReadFile(filepath.Join(fixture, "runtime_fixture.py"))
			must(t, err)
			must(t, os.WriteFile(filepath.Join(control, "bin", "cozy-runtime"), append([]byte("#!"+python+"\n"), launcher...), 0700)) //cozy:allow records real Runtime worker preparation in the owned fixture launcher
			audit := filepath.Join(root, "serving_fixture_audit.jsonl")
			must(t, os.Symlink(audit, filepath.Join(control, "bin", "serving_fixture_audit.jsonl")))
			path := filepath.Join(control, "bin")
			for _, value := range childEnv(t, root) {
				if strings.HasPrefix(value, "PATH=") {
					path += string(os.PathListSeparator) + strings.TrimPrefix(value, "PATH=")
				}
			}
			t.Cleanup(func() {
				if _, err := os.Stat(filepath.Join(root, "creator.sqlite")); err == nil {
					compositionDown(t, root, path)
				} else if !os.IsNotExist(err) {
					t.Errorf("read fixture home before teardown: %v", err)
				} else {
					for _, event := range preparationEvents(t, audit) {
						if event.Method != "WorkerProcess" {
							continue
						}
						raw, err := os.ReadFile(filepath.Join("/proc", strconv.Itoa(event.WorkerPID), "stat"))
						if os.IsNotExist(err) {
							continue
						}
						if err != nil {
							t.Errorf("read owned worker after early refusal: %v", err)
							continue
						}
						fields := strings.Fields(string(raw)[strings.LastIndex(string(raw), ")")+1:])
						if len(fields) < 20 || fields[19] == event.Birth {
							t.Errorf("metadata refusal left owned worker %d alive before database creation", event.WorkerPID)
						}
					}
				}
				if t.Failed() {
					t.Log("local serving preparation evidence retained", root)
				} else {
					// Immutable Runtime generations are read-only. Only this test's
					// verified-stopped home is made removable, without following links.
					err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
						if err == nil && entry.IsDir() {
							return os.Chmod(path, 0700)
						}
						return err
					})
					if err == nil {
						err = os.RemoveAll(root)
					}
					if err != nil {
						t.Errorf("remove stopped fixture home: %v", err)
					}
				}
			})
			project := t.TempDir()
			module, err := os.ReadFile(filepath.Join(fixture, "serving_fixture.py"))
			must(t, err)
			if canary {
				module = bytes.Replace(module, []byte("REFUSE_INITIALIZATION = False"), []byte("REFUSE_INITIALIZATION = True"), 1)
			}
			must(t, os.WriteFile(filepath.Join(project, "serving_fixture.py"), module, 0600))
			packageTOML, err := os.ReadFile(filepath.Join(fixture, "package.toml"))
			must(t, err)
			packageTOML = append(packageTOML, []byte("\n[bindings.\"generate.models.model\"]\nmodel=\"proof/ordered\"\nrelease=\"1.0.0\"\nlane=\"bf16\"\n")...)
			must(t, os.WriteFile(filepath.Join(project, "package.toml"), packageTOML, 0600))
			metadata := fmt.Sprintf(`[project]
name="cozy-serving-preparation-fixture"
version="0.0.1"
requires-python=">=3.12,<3.13"
dependencies=["cozy-runtime==%s","torch>=2.13,<3"]
[project.entry-points."cozy.application"]
default="serving_fixture:app"
[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
only-include=["serving_fixture.py"]
[tool.uv.sources]
cozy-runtime={path=%q}
%s
`, version, wheel, servingTorchIndex())
			must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(metadata), 0600))
			uv("lock", "--project", project, "--no-progress")
			pkg := "local/cozy-serving-preparation-fixture"
			if code, out := runCozyPath(t, root, path, "package", "install", project, "--editable", "--json"); code != 0 {
				t.Fatalf("metadata install initialized the model or refused: %d %s", code, out)
			}
			if code, out := runCozyPath(t, root, path, "run", pkg+"/generate", "--describe", "--json"); code != 0 {
				t.Fatalf("static description initialized the model or refused: %d %s", code, out)
			}
			for _, event := range preparationEvents(t, audit) {
				if event.Method != "WorkerProcess" {
					t.Fatalf("metadata intake triggered worker preparation: %+v", event)
				}
			}
			before := activeInstall(t, root, pkg)
			if before.PlacementSetDigest != "" {
				t.Fatal("metadata intake invented a serving placement")
			}
			model := "model.model=proof/ordered@1.0.0/bf16#" + seed.Manifest
			code, out := runCozyPath(t, root, path, "run", pkg+"/generate", "value=3", model, "--await", "--json")
			events := preparationEvents(t, audit)
			var prepared *servingPreparationEvent
			for i := range events {
				if events[i].Method == "PreparePrivatePlacement" {
					prepared = &events[i]
				}
			}
			if prepared == nil || prepared.WorkerPID <= 0 || prepared.OperationID == "" {
				t.Fatalf("run did not reach real worker placement preparation: code=%d out=%s events=%+v", code, out, events)
			}
			if canary {
				match := regexp.MustCompile(`local-serving-init-canary pid=(\d+) ppid=(\d+) device=meta`).FindStringSubmatch(prepared.Error)
				if code == 0 || len(match) != 3 {
					t.Fatalf("initialization was not a typed worker refusal: code=%d out=%s event=%+v", code, out, prepared)
				}
				parent, err := strconv.Atoi(match[2])
				must(t, err)
				if parent != prepared.WorkerPID {
					t.Fatalf("constructor parent %d is not preparation worker %d", parent, prepared.WorkerPID)
				}
				t.Logf("actual worker %d reported constructor child %s on meta; metadata intake did not initialize it", prepared.WorkerPID, match[1])
				return
			}
			if prepared.Error != "" {
				t.Fatalf("normal model preparation refused: %+v", prepared)
			}
			raw, err := base64.StdEncoding.DecodeString(prepared.Placement)
			must(t, err)
			digest, err := canonical.Spell(canonical.Digest(raw))
			must(t, err)
			if digest != "sha256:"+prepared.Digest {
				t.Fatal("actual preparation reply has a different digest")
			}
			set, err := canonical.Read(raw, &pb.PlacementSet{})
			must(t, err)
			placements := set.List("placements")
			if len(placements) != 1 {
				t.Fatalf("preparation returned %d placements", len(placements))
			}
			entrypoints := placements[0].List("entrypoints")
			if len(entrypoints) != 1 || entrypoints[0].Str("name") != "generate" {
				t.Fatal("preparation changed the fixture's entrypoint")
			}
			slots := entrypoints[0].List("slots")
			if len(slots) != 1 {
				t.Fatalf("preparation returned %d model slots", len(slots))
			}
			var components []string
			for _, component := range slots[0].List("components") {
				components = append(components, component.Str("component"))
			}
			if !slices.Equal(components, []string{"zeta", "alpha"}) {
				t.Fatalf("worker lost actual construction order: %v; header=%v", components, seed.HeaderComponents)
			}
			after := activeInstall(t, root, pkg)
			retained, err := os.ReadFile(filepath.Join(after.Dir, "artifact-cache", prepared.Digest))
			must(t, err)
			if !bytes.Equal(retained, raw) {
				t.Fatal("owner retained different placement bytes")
			}
			var acquired *servingPreparationEvent
			for i := range events {
				if events[i].Method == "Acquire" {
					acquired = &events[i]
				}
			}
			if acquired == nil || acquired.Receipt == "" || acquired.Environment == acquired.DependencyEnvironment ||
				!strings.HasPrefix(acquired.Environment, filepath.Join(after.Dir, "worker-environments", "contents")+string(filepath.Separator)) {
				t.Fatalf("activation did not retain the captured immutable code environment: %+v", acquired)
			}
			store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
			fatal(t, problem)
			defer store.Close()
			request, problem := store.RequestByReference("1")
			fatal(t, problem)
			if request == nil || request.PlanID != entrypoints[0].Str("entrypoint_binding_digest") {
				t.Fatalf("request did not select the real returned binding: %+v", request)
			}
			if code != 0 {
				t.Fatalf("prepared model did not complete normal activation/inference: %d %s", code, out)
			}
			if request.State != "succeeded" {
				t.Fatalf("successful CLI result has request state %s", request.State)
			}
			if !regexp.MustCompile(`"value":12(?:\.0)?[,}]`).MatchString(out) {
				t.Fatalf("normal inference did not read the actual filled tensors: %s", out)
			}
			t.Logf("actual worker %d retained components %v and immutable environment %s; inference returned 12", prepared.WorkerPID, components, acquired.Receipt)
		})
	}
}
