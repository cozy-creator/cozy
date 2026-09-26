package producttest

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

// The fixture seeds an exact published install; both submissions use ordinary
// cozy run and the same local Runtime used by unpublished scripts.
func TestPublishedMachineActualLocalAndNewRootAfterRestart(t *testing.T) {
	if *publishedMachineFixture == "" || *privateChildRuntimeWheel == "" {
		t.Skip("requires an exact published wheel/interface and Runtime wheel")
	}
	var fixture struct {
		Package   string          `json:"package"`
		Release   string          `json:"release"`
		Wheel     string          `json:"wheel"`
		Interface json.RawMessage `json:"interface"`
	}
	raw, err := os.ReadFile(*publishedMachineFixture)
	must(t, err)
	must(t, json.Unmarshal(raw, &fixture))
	iface, problem := launch.DecodePackageInterface(fixture.Interface)
	fatal(t, problem)
	wheel, err := os.ReadFile(fixture.Wheel)
	must(t, err)
	wheelDigest, err := canonical.Spell(canonical.Digest(wheel))
	must(t, err)
	wheelURL := publishedFixtureWheel(t, wheel, filepath.Base(fixture.Wheel))
	control := filepath.Join(t.TempDir(), "control")
	for _, args := range [][]string{{"venv", control, "--python", "3.12"},
		{"pip", "install", "--python", filepath.Join(control, "bin", "python"), *privateChildRuntimeWheel}} {
		out, err := exec.Command("uv", args...).CombinedOutput()
		if err != nil {
			t.Fatalf("published Runtime SDK: %v\n%s", err, out)
		}
	}
	root, err := os.MkdirTemp("", "cozy-pub-local-")
	must(t, err)
	path := filepath.Join(control, "bin")
	for _, value := range childEnv(t, root) {
		if strings.HasPrefix(value, "PATH=") {
			path += string(os.PathListSeparator) + strings.TrimPrefix(value, "PATH=")
		}
	}
	t.Cleanup(func() {
		compositionDown(t, root, path)
		if t.Failed() {
			t.Log("published local evidence retained", root)
		} else {
			must(t, removeAllForce(root))
		}
	})
	installDir := filepath.Join(root, "installs", "published-local-proof")
	must(t, os.MkdirAll(filepath.Join(installDir, "source"), 0700))
	must(t, os.MkdirAll(filepath.Dir(launch.PackageInterfacePath(installDir)), 0700))
	must(t, os.WriteFile(launch.PackageInterfacePath(installDir), iface.Raw, 0600))
	must(t, os.Symlink(control, filepath.Join(installDir, "venv")))
	_, distribution, _ := strings.Cut(fixture.Package, "/")
	locked := []byte(fmt.Sprintf("--index-url https://pypi.org/simple\n%s @ %s --hash=%s\n", distribution, wheelURL, wheelDigest))
	must(t, os.WriteFile(filepath.Join(installDir, "locked-requirements.txt"), locked, 0600))
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	_, problem = store.Activate(records.PackageInstall{ID: "published-local-proof", Package: fixture.Package, Major: 1,
		Version: fixture.Release, SourceKind: "tensorhub", SourceRef: fixture.Package + "@" + fixture.Release,
		Dir: installDir, Platform: "linux-x86_64",
		Runtime: runtimeFixtureVersion(t, *privateChildRuntimeWheel), Closure: "cozy-runtime>=0.17.2"})
	fatal(t, problem)
	defer store.Close()
	var build string
	for index := range 2 {
		if index == 1 && !reapMachineRuntimeRoot(root) {
			t.Fatal("owned local Runtime did not stop for replacement")
		}
		key := fmt.Sprintf("published-local-%d", index)
		code, out := runCozyPath(t, root, path, "run", fixture.Package+"/main", "--await", "--json", "--idempotency-key", key)
		if code != 0 || !strings.Contains(out, `"value":8`) {
			t.Fatalf("published local root %d [%d]: %s\n%s", index, code, out, productWorkerLogs(root))
		}
		request, problem := store.RequestByIdempotencyKey(key)
		fatal(t, problem)
		if request == nil || request.State != "succeeded" || request.LocalInstallationID != "" || request.Release != fixture.Release || request.Rental {
			t.Fatalf("published local request changed origin: %+v", request)
		}
		link, problem := store.MachineExecution(request.ID)
		fatal(t, problem)
		if link == nil || link.MachineID != "local" || !link.Collected || len(link.Receipt) == 0 {
			t.Fatal("local root has no collected Runtime receipt")
		}
		var submitted pb.MachineExecutionSubmit
		must(t, proto.Unmarshal(link.Submission, &submitted))
		var capture pb.MachineExecutionCapture
		must(t, canonical.Unmarshal(submitted.CaptureCanonicalBytes, &capture))
		if len(capture.Revisions) != 0 || len(capture.PublishedRevisions) != 1 {
			t.Fatal("local published root was recast as private code")
		}
		_, identity, err := canonical.Identity(capture.PublishedRevisions[0].Environment)
		must(t, err)
		current, err := canonical.Spell(identity)
		must(t, err)
		if !bytes.Equal(identity, capture.RootInstallationId) || submitted.PreparedState.GetJob().InstallationId != current || (build != "" && current != build) {
			t.Fatal("local published Environment identity changed")
		}
		build = current
		attempts, problem := store.Attempts(request.ID)
		fatal(t, problem)
		children, problem := store.Children(request.ID)
		fatal(t, problem)
		if len(attempts) != 0 || len(children) != 0 {
			t.Fatal("Creator owns local published attempts or children")
		}
		t.Logf("local published root %s accepted and collected, Environment=%s: %s", request.ID, build, out)
	}
}
