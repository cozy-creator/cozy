package producttest

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/machines"
)

// oldestRuntime is the oldest released Runtime this Creator serves (the run output log, 0.18.73).
const oldestRuntime = "0.18.73"

// publishedWheel is one distribution's released linux x86_64 wheel from PyPI, verified by its
// digest and kept for the run: the bytes `cozy machine install` puts on a machine.
func publishedWheel(t *testing.T, distribution, version string) string {
	t.Helper()
	response, err := http.Get("https://pypi.org/pypi/" + distribution + "/" + version + "/json")
	if err != nil {
		t.Skipf("PyPI unreachable from this runner: %v", err)
	}
	defer response.Body.Close()
	var release struct {
		URLs []struct {
			Filename string            `json:"filename"`
			URL      string            `json:"url"`
			Digests  map[string]string `json:"digests"`
		} `json:"urls"`
	}
	must(t, json.NewDecoder(response.Body).Decode(&release))
	for _, file := range release.URLs {
		if !strings.HasSuffix(file.Filename, "-cp312-abi3-manylinux_2_28_x86_64.whl") &&
			!strings.HasSuffix(file.Filename, "-manylinux_2_17_x86_64.manylinux2014_x86_64.whl") {
			continue
		}
		path := filepath.Join(scratchBase, "published-wheels", file.Filename)
		if raw, err := os.ReadFile(path); err == nil && sha256Hex(raw) == file.Digests["sha256"] {
			return path
		}
		download, err := http.Get(file.URL)
		must(t, err)
		raw, err := io.ReadAll(download.Body)
		download.Body.Close()
		must(t, err)
		if sha256Hex(raw) != file.Digests["sha256"] {
			t.Fatalf("%s does not match its published digest", file.Filename)
		}
		must(t, os.MkdirAll(filepath.Dir(path), 0o700))
		must(t, os.WriteFile(path, raw, 0o600))
		return path
	}
	t.Fatalf("%s %s publishes no linux x86_64 wheel", distribution, version)
	return ""
}

func sha256Hex(raw []byte) string {
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:])
}

// Version skew is allowed: this Creator runs unpublished and published packages, one of them
// with an org-relative Model default, on machines running the oldest Runtime it serves, both
// this computer's and a rental, with the real Host. The packages lock the published Runtime.
func TestTheOldestServedRuntimeRunsUnpublishedAndPublishedPackages(t *testing.T) {
	if *machineHostBinary == "" {
		t.Skip("requires -machine-host: the pod-supervisor both machines run")
	}
	source := machines.Source{Host: *machineHostBinary, RuntimeWheel: publishedWheel(t, hostruntime.Distribution, oldestRuntime),
		TensorFSWheel: publishedWheel(t, "tensorfs", "0.3.74")}
	h, root, _, _ := parityMachinesOn(t, source)
	python := filepath.Join(root, "machine", "root", "opt", "cozy", "python", "bin", "python")
	var resolved string
	for _, store := range []string{filepath.Join(root, "tensorfs"), filepath.Join(h.provider, "var", "lib", "tensorfs")} {
		out, err := exec.Command(python, "-I", "-c", seedCheckpoint, store).CombinedOutput()
		if err != nil {
			t.Fatalf("seeding %s: %v\n%s", store, err, out)
		}
		resolved = strings.TrimSpace(string(out))
	}
	h.mux.HandleFunc("GET /v1/accounts/current", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"name":"proof"}`))
	})
	doors := h.worker.Config.Handler
	h.worker.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models/resolve" && r.URL.Query().Get("ref") == "proof/probe@1.0.0" && r.URL.Query().Get("lane") == "bf16" {
			var body map[string]any
			must(t, json.Unmarshal([]byte(resolved), &body))
			body["model"], body["release"], body["lane"] = "proof/probe", "1.0.0", "bf16"
			_ = json.NewEncoder(w).Encode(body)
			return
		}
		doors.ServeHTTP(w, r)
	})

	released := machines.Source{}
	if code, out := runCozy(t, root, "package", "install", parityProjectOn(t, released), "--editable"); code != 0 {
		t.Fatalf("editable install [exit %d]\n%s", code, out)
	}
	probe := probeProjectOn(t, released, "probe@1.0.0/bf16")
	pyproject := filepath.Join(probe, "pyproject.toml")
	raw, err := os.ReadFile(pyproject)
	must(t, err)
	must(t, os.WriteFile(pyproject, []byte(strings.Replace(string(raw), `name="machine-parity"`, `name="skew-probe"`, 1)), 0o600))
	if out, err := exec.Command("uv", "lock", "--project", probe).CombinedOutput(); err != nil {
		t.Fatalf("locking the probe: %v\n%s", err, out)
	}
	if code, out := runCozy(t, root, "package", "install", probe, "--editable"); code != 0 {
		t.Fatalf("editable probe install [exit %d]\n%s", code, out)
	}
	publishParityRelease(t, h, root, parityProjectOn(t, released))
	for venue, args := range map[string][]string{"local": nil, "rental": {"--rental=tessa"}} {
		for _, call := range []struct{ target, input, want string }{
			{"local/machine-parity/add", "value=41", `"value":42`},
			{"local/machine-parity/echo", "value=41", `"value":82`},
			{"local/skew-probe/touch", "value=1", `"value":2`},
			{parityPublished + "/add", "value=41", `"value":42`},
		} {
			code, out := runCozy(t, root, append([]string{"run", call.target, call.input, "--await", "--json"}, args...)...)
			if code != 0 || !strings.Contains(out, call.want) {
				t.Fatalf("%s on the %s %s machine [exit %d]\n%s", call.target, oldestRuntime, venue, code, out)
			}
		}
	}
}
