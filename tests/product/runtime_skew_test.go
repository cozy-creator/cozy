package producttest

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/machines"
)

// retiredRuntime predates the wheel-owned agent and stable bootstrap contracts.
const retiredRuntime = "0.18.73"

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

// A retired SDK is not a supported machine installation just because its wire
// range overlaps. The real CLI refuses missing bootstrap capability before any
// machine process or execution exists; older read/cancel contracts stay separate.
func TestRetiredRuntimeBootstrapRefusesBeforeAnyWork(t *testing.T) {
	if _, err := exec.LookPath("uv"); err != nil {
		t.Skip("uv lays out the isolated machine")
	}
	runtime := publishedWheel(t, hostruntime.Distribution, retiredRuntime)
	tensorfs := publishedWheel(t, "tensorfs", "0.3.74")
	root := t.TempDir()
	// Do not let the test helper auto-provision a current fixture into this
	// intentionally empty machine root; this command tests first installation.
	must(t, os.Mkdir(filepath.Join(root, "machine"), 0700))
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
	defer cancel()
	command := exec.CommandContext(ctx, cozyBin, "machine", "install", "--runtime-wheel", runtime, "--tensorfs-wheel", tensorfs, "--json")
	command.Env = childEnv(t, root)
	out, err := command.CombinedOutput()
	if ctx.Err() != nil || err == nil || !strings.Contains(string(out), "machine.agent_update_required") || !strings.Contains(string(out), machines.BootstrapCapability) {
		t.Fatalf("retired bootstrap did not refuse its missing capability: %v %s", err, out)
	}
	for _, name := range []string{"installed.json", "agent.json", "owner.pem"} {
		if _, err := os.Stat(filepath.Join(root, "machine", name)); !os.IsNotExist(err) {
			t.Fatalf("refused bootstrap created %s", name)
		}
	}
	path := filepath.Join(root, "creator.sqlite")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return
	}
	db, err := sql.Open("sqlite", "file:"+path+"?mode=ro")
	must(t, err)
	defer db.Close()
	var tables, accepted int
	must(t, db.QueryRow("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='machine_executions'").Scan(&tables))
	if tables != 0 {
		must(t, db.QueryRow("SELECT count(*) FROM machine_executions WHERE length(submission)>0 OR length(receipt)>0").Scan(&accepted))
	}
	if accepted != 0 {
		t.Fatal("unsupported bootstrap froze or accepted work")
	}
}
