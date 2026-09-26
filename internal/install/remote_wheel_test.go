package install

import (
	"archive/zip"
	"context"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestRemoteWheelRetainsCaptureWithoutLocalEnvironment(t *testing.T) {
	if _, problem := config.Load(); problem != nil {
		t.Fatal(problem)
	}
	root := t.TempDir()
	path := filepath.Join(root, "remote_callable-1.0.0-py3-none-any.whl")
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	archive := zip.NewWriter(file)
	for name, body := range map[string]string{
		"remote_callable.py":                               "from cozy_runtime.author import App\napp=App()\n",
		"remote_callable-1.0.0.dist-info/METADATA":         "Metadata-Version: 2.1\nName: remote-callable\nVersion: 1.0.0\nRequires-Python: >=3.12\nRequires-Dist: cozy-runtime>=0.18.0\n",
		"remote_callable-1.0.0.dist-info/WHEEL":            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
		"remote_callable-1.0.0.dist-info/entry_points.txt": "[cozy.application]\ndefault=remote_callable:app\n",
		"remote_callable-1.0.0.dist-info/RECORD":           "",
	} {
		member, err := archive.Create(name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := io.WriteString(member, body); err != nil {
			t.Fatal(err)
		}
	}
	if err := archive.Close(); err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	dependency, problem := packagepublish.CaptureDependency(path)
	if problem != nil {
		t.Fatal(problem)
	}
	surface, problem := launch.DecodePackageInterface([]byte(`{"format":"cozy.package.interface/1","application":"remote_callable:app","entrypoints":[],"jobs":[]}`))
	if problem != nil {
		t.Fatal(problem)
	}
	layout, problem := home.Open(filepath.Join(root, "records"))
	if problem != nil {
		t.Fatal(problem)
	}
	store, problem := records.Open(layout.DB)
	if problem != nil {
		t.Fatal(problem)
	}
	defer store.Close()
	dependencies := map[string]packagepublish.CapturedDependency{"remote-callable": dependency}
	selected := packagepublish.RequirementSelection{RequiresPython: ">=3.12", Requirements: []string{"cozy-runtime>=0.18.0"}}
	if _, problem := CaptureRemoteWheel(context.Background(), layout, store, "remote-callable", "3.12.12", nil, dependencies, surface, selected); problem == nil {
		t.Fatal("remote wheel omitted its required Runtime closure")
	}
	dependencies["cozy-runtime"] = packagepublish.CapturedDependency{Name: "cozy-runtime", Version: "0.18.30", Requirement: "cozy-runtime==0.18.30"}
	result, problem := CaptureRemoteWheel(context.Background(), layout, store, "remote-callable", "3.12.12", nil, dependencies, surface, selected)
	if problem != nil {
		t.Fatal(problem)
	}
	if !result.RemoteSnapshot {
		t.Fatal("wheel did not use remote capture")
	}
	if _, err := os.Stat(filepath.Join(result.Install.Dir, "venv")); !os.IsNotExist(err) {
		t.Fatalf("wheel constructed a local venv: %v", err)
	}
	if _, problem := InstalledRequirements(context.Background(), result.Install); problem != nil {
		t.Fatal(problem)
	}
	if _, problem := localpackage.StageWheels(layout, result.Install, surface.Raw, []string{result.CapturedProjectWheel}, []byte("cozy-runtime==0.18.30\n")); problem != nil {
		t.Fatal(problem)
	}
}
