package producttest

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
)

// These are real CLI processes with a retained package interface. The GPU and
// empty-store tools are stand-ins: this proves admission and lane selection, not
// GPU compatibility or TensorFS acquisition. The Hub refuses at model resolution
// so no model bytes, worker process, or paid rental can be acquired.
type admissionProbe struct {
	mu       sync.Mutex
	requests []string
	lanes    []string
}

func admissionInterface(t *testing.T, optionalPrompt, requiredImages bool) []byte {
	t.Helper()
	var doc map[string]any
	must(t, json.Unmarshal([]byte(declaredAssetsInterface), &doc))
	ep := doc["entrypoints"].([]any)[0].(map[string]any)
	ep["name"] = "generate"
	ep["models"] = []any{map[string]any{"class": "H3", "path": ladderSlot, "component_use": map[string]any{}}}
	fields := ep["request"].(map[string]any)["fields"].([]any)
	prompt := fields[0].(map[string]any)
	prompt["constraints"] = map[string]any{"min_length": 1}
	if optionalPrompt {
		prompt["wire"] = "optional"
	}
	if !requiredImages {
		delete(fields[1].(map[string]any)["constraints"].(map[string]any), "min_length")
	}
	ep["request"].(map[string]any)["fields"] = append(fields[:2], map[string]any{
		"name": "steps", "type": "int", "wire": "optional", "constraints": map[string]any{"ge": 1, "le": 50},
	})
	raw, err := json.Marshal(doc)
	must(t, err)
	return raw
}

func admissionRoot(t *testing.T, iface []byte, gpu string, offline bool) (root, path, activity string, probe *admissionProbe) {
	t.Helper()
	h := newLadderHub(t)
	row := goodLadder()
	row.Ladder = row.Ladder[:2] // H100 and B200, deliberately no catch-all.
	h.bind(row)
	probe = &admissionProbe{}
	parsed, problem := launch.DecodePackageInterface(iface)
	fatal(t, problem)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		probe.mu.Lock()
		probe.requests = append(probe.requests, r.Method+" "+r.URL.Path)
		if r.URL.Path == "/v1/models/resolve" {
			probe.lanes = append(probe.lanes, r.URL.Query().Get("lane"))
		}
		probe.mu.Unlock()
		if offline || r.URL.Path == "/v1/models/resolve" {
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte(`{"error":{"code":"proof.stop_before_acquisition","message":"proof stopped before model acquisition"}}`))
			return
		}
		if r.URL.Path == "/v1/packages/proof/h3/releases/1.0.0" {
			var detail hub.PackageReleaseDetail
			detail.PackageInterface = iface
			detail.Release.Release = "1.0.0"
			detail.Release.PackageInterfaceDigest = assessmentDigest(parsed.Raw)
			detail.Release.PackageInterfaceLength = int64(len(iface))
			detail.ExecutionRequirements = []string{"cozy-runtime>=0.2.25", "torch<3,>=2.13"}
			_ = json.NewEncoder(w).Encode(detail)
			return
		}
		h.server.Config.Handler.ServeHTTP(w, r)
	}))
	t.Cleanup(server.Close)
	root, path = t.TempDir(), t.TempDir()
	t.Cleanup(func() { _, _ = runCozy(t, root, "down", "--all") })
	activity = filepath.Join(root, "tool-activity")
	sshKeygen, err := exec.LookPath("ssh-keygen")
	must(t, err)
	must(t, os.Symlink(sshKeygen, filepath.Join(path, "ssh-keygen")))
	quote := func(s string) string { return "'" + strings.ReplaceAll(s, "'", "'\"'\"'") + "'" }
	// Both stand-ins append to one observable trace. An invalid request must not
	// even ensure the empty store or query the accelerator.
	tensorfs := filepath.Join(path, "tfs")
	must(t, os.WriteFile(tensorfs, []byte("#!/bin/sh\nprintf 'tfs\\n' >> "+quote(activity)+"\n"+
		"case \"$1 $2\" in\n'store ensure') exit 0;;\n'repo list') : > \"$5\"; exit 0;;\nesac\nexit 97\n"), 0700))
	must(t, os.WriteFile(filepath.Join(path, "nvidia-smi"), []byte("#!/bin/sh\nprintf 'gpu\\n' >> "+quote(activity)+"\n"+
		"if [ \"$#\" -eq 0 ]; then printf 'CUDA Version: 13.0\\n'; else printf '%s\\n' "+
		quote("0, "+gpu+", 8192, 8192, 580.82.09, 8.9")+"; fi\n"), 0700))
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(fmt.Sprintf(
		"tensorhub_url: %s\ntensorhub_token: ladder-test\ntfs: %s\ndaemon:\n  idle_shutdown_s: 0\n", server.URL, tensorfs)), 0600))
	installDir := filepath.Join(root, "installs", "inst-admission")
	must(t, os.MkdirAll(filepath.Dir(launch.PackageInterfacePath(installDir)), 0700))
	must(t, os.WriteFile(launch.PackageInterfacePath(installDir), iface, 0600))
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	_, problem = store.Activate(records.PackageInstall{ID: "inst-admission", Package: ladderPackage,
		Major: 1, Version: "1.0.0", SourceKind: "tensorhub", Dir: installDir,
		Platform: "linux-x86"})
	fatal(t, problem)
	return root, path, activity, probe
}

// The shared child-env policy freezes once per test binary; impose each fixture's
// two tool locations explicitly so one test root never inherits another's tools.
func runAdmissionCLI(t *testing.T, root, path string, args ...string) (int, string) {
	t.Helper()
	cmd := exec.Command("/usr/bin/nice", append([]string{"-n", "19", cozyBin}, args...)...)
	cmd.Env = childEnv(t, root, "PATH="+path, "COZY_TFS="+filepath.Join(path, "tfs"))
	out, err := cmd.CombinedOutput()
	if err != nil && cmd.ProcessState == nil {
		t.Fatal(err)
	}
	return cmd.ProcessState.ExitCode(), string(out)
}

// daemonFleetHousekeeping is the traffic the DAEMON generates for its own reasons,
// which these arms are not about. `cozy run` dials the daemon, and a daemon asks the
// hub what rentals this account owns whether or not it holds any records of its own
// (cl-199) — that reverse reconcile is the only way a pod nobody recorded is ever
// noticed, so it cannot be conditioned on the local fleet being non-empty. It is the
// same class as the per-rental reconcile the daemon already runs, which this fixture
// simply never saw because it holds no rentals.
//
// What these arms assert is unchanged: nothing the INVALID INPUT touches reaches the
// hub, the store, or the GPU.
func daemonFleetHousekeeping(request string) bool { return request == "GET /v1/rentals" }

func (p *admissionProbe) snapshot() (requests, lanes []string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]string(nil), p.requests...), append([]string(nil), p.lanes...)
}

func assertAdmissionDidNotSubmit(t *testing.T, root, key string) {
	t.Helper()
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	row, problem := store.RequestByIdempotencyKey(key)
	fatal(t, problem)
	if row != nil {
		t.Fatalf("refused input was submitted: %+v", row)
	}
}

func TestRunValidatesArgumentsBeforeModelResolution(t *testing.T) {
	for _, test := range []struct {
		name, field string
		args        []string
		offline     bool
		images      bool
	}{
		{name: "missing prompt on unmatched GPU", field: "prompt"},
		{name: "missing prompt with offline Hub", field: "prompt", offline: true},
		{name: "empty prompt", field: "prompt", args: []string{"prompt="}},
		{name: "wrong prompt type", field: "prompt", args: []string{"prompt:=7"}},
		{name: "invalid steps", field: "steps", args: []string{"prompt=hello", "steps=0"}},
		{name: "missing image collection", field: "assets", args: []string{"prompt=hello"}, images: true},
		{name: "rented missing prompt", field: "prompt", args: []string{"--rental-only"}},
		{name: "named rental missing prompt", field: "prompt", args: []string{"--rental=not-resolved"}},
		{name: "named rental empty prompt", field: "prompt", args: []string{"--rental=not-resolved", "prompt="}},
	} {
		t.Run(test.name, func(t *testing.T) {
			root, path, activity, probe := admissionRoot(t, admissionInterface(t, false, test.images),
				"NVIDIA GeForce RTX 4070 Laptop GPU", test.offline)
			args := append([]string{"run", ladderPackage + "/generate", "--json", "--idempotency-key", "invalid-arguments"}, test.args...)
			code, out := runAdmissionCLI(t, root, path, args...)
			if code != 1 || !strings.Contains(out, `"code":"request_payload_invalid"`) || !strings.Contains(out, test.field) {
				t.Errorf("argument refusal lost to model resolution [exit %d]: %s", code, out)
			}
			if strings.Contains(out, "Resolving model") {
				t.Errorf("invalid arguments began model resolution: %s", out)
			}
			requests, _ := probe.snapshot()
			for _, request := range requests {
				if request != "GET /v1/packages/proof/h3" && request != "GET /v1/packages/proof/h3/releases/1.0.0" &&
					!daemonFleetHousekeeping(request) {
					t.Errorf("invalid input reached %s", request)
				}
			}
			if _, err := os.Stat(activity); !os.IsNotExist(err) {
				t.Errorf("invalid input opened the store or probed GPUs: %s", tail(activity))
			}
			assertAdmissionDidNotSubmit(t, root, "invalid-arguments")
		})
	}
}

func TestBareRunReportsRequiredArgumentsAndFullInterface(t *testing.T) {
	root, path, activity, probe := admissionRoot(t, admissionInterface(t, false, false), "unmatched GPU", true)
	code, out := runAdmissionCLI(t, root, path, "run", ladderPackage+"/generate")
	for _, want := range []string{"Error: provide required arguments:", "prompt", "prompt: str", "assets:", "steps: int", "(optional)"} {
		if code != 1 || !strings.Contains(out, want) {
			t.Errorf("bare run omitted %q [exit %d]: %s", want, code, out)
		}
	}
	requests, _ := probe.snapshot()
	for _, request := range requests {
		if !daemonFleetHousekeeping(request) {
			t.Errorf("rendering argument help dialed Hub: %v", requests)
			break
		}
	}
	if _, err := os.Stat(activity); !os.IsNotExist(err) {
		t.Errorf("rendering argument help opened tools: %s", tail(activity))
	}
}

func TestLocalRunLadderIsAPreferenceNotAnAllowlist(t *testing.T) {
	for _, test := range []struct {
		name, gpu, want string
		args            []string
		optionalPrompt  bool
	}{
		{name: "unmatched GPU uses first declared lane", gpu: "NVIDIA GeForce RTX 4070 Laptop GPU", want: ladderLane, args: []string{"prompt=hello"}},
		{name: "matching GPU uses its rung", gpu: "NVIDIA B200", want: "mxfp8-adaln-pruned", args: []string{"prompt=hello"}},
		{name: "explicit lane overrides matching rung", gpu: "NVIDIA B200", want: "bf16-full", args: []string{"prompt=hello", "model.model=" + ladderModel + "@" + ladderRelease + "/bf16-full"}},
		{name: "no mandatory arguments proceeds", gpu: "NVIDIA B200", want: "mxfp8-adaln-pruned", optionalPrompt: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			root, path, _, probe := admissionRoot(t, admissionInterface(t, test.optionalPrompt, false), test.gpu, false)
			args := append([]string{"run", ladderPackage + "/generate", "--json", "--idempotency-key", "local-selection"}, test.args...)
			code, out := runAdmissionCLI(t, root, path, args...)
			_, lanes := probe.snapshot()
			if code == 0 || !strings.Contains(out, "proof stopped before model acquisition") || len(lanes) != 1 || lanes[0] != test.want {
				t.Fatalf("local selection did not reach acquisition with lane %s: lanes=%v exit=%d %s", test.want, lanes, code, out)
			}
			assertAdmissionDidNotSubmit(t, root, "local-selection")
		})
	}
}
