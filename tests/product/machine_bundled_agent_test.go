package producttest

import (
	"archive/zip"
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

	"github.com/cozy-creator/cozy/internal/machines"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

type noAgentDiscovery struct{ t *testing.T }

func (n noAgentDiscovery) RoundTrip(r *http.Request) (*http.Response, error) {
	n.t.Errorf("bundled/incumbent agent unexpectedly requested %s", r.URL)
	return nil, fmt.Errorf("release discovery disabled in this test")
}

func bundledTestAgent(t *testing.T, version string, capabilities ...string) string {
	t.Helper()
	dir := t.TempDir()
	if len(capabilities) == 0 {
		capabilities = []string{machines.HubAccessCapability, machines.RuntimeUpdateCapability, machines.BootstrapCapability}
	}
	writeInstallFile(t, filepath.Join(dir, "go.mod"), []byte("module "+machines.AgentModule+"\n\ngo 1.26\n"), 0600)
	identity, err := json.Marshal(map[string]any{"name": "cozy-machine", "version": version,
		"wire_minor": pb.WireMinor, "minimum_wire_minor": pb.MinCompatibleWireMinor, "capabilities": capabilities})
	must(t, err)
	writeInstallFile(t, filepath.Join(dir, "main.go"), []byte(fmt.Sprintf("package main\nimport \"fmt\"\nfunc main(){fmt.Print(%q)}\n", identity)), 0600)
	path := filepath.Join(dir, "cozy-machine")
	command := exec.Command("go", "build", "-o", path, ".")
	command.Dir = dir
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("build fixture ELF: %v\n%s", err, output)
	}
	return path
}

func bundledRuntimeWheel(t *testing.T, version, agent string) string {
	t.Helper()
	source := installWheel(t, "cozy_runtime", version, "cozy-runtime-worker")
	r, err := zip.OpenReader(source)
	must(t, err)
	defer r.Close()
	path := filepath.Join(t.TempDir(), filepath.Base(source))
	out, err := os.Create(path)
	must(t, err)
	w := zip.NewWriter(out)
	script := "cozy_runtime-" + version + ".data/scripts/cozy-machine"
	for _, file := range r.File {
		stream, err := file.Open()
		must(t, err)
		data, err := io.ReadAll(stream)
		must(t, err)
		stream.Close()
		if strings.HasSuffix(file.Name, ".dist-info/RECORD") {
			data = append(data, []byte(script+",,\n")...)
		}
		entry, err := w.CreateHeader(&file.FileHeader)
		must(t, err)
		_, err = entry.Write(data)
		must(t, err)
	}
	header := &zip.FileHeader{Name: script, Method: zip.Deflate}
	header.SetMode(0755)
	entry, err := w.CreateHeader(header)
	must(t, err)
	data, err := os.ReadFile(agent)
	must(t, err)
	_, err = entry.Write(data)
	must(t, err)
	must(t, w.Close())
	must(t, out.Close())
	return path
}

func TestMachineDefaultUsesWheelAgentAndRejectsUnbundledWheels(t *testing.T) {
	uv := offlineInstallerUV(t)
	t.Setenv("UV_NO_INDEX", "1")
	index := t.TempDir()
	t.Setenv("UV_FIND_LINKS", index)
	oldClient := http.DefaultClient
	http.DefaultClient = &http.Client{Transport: noAgentDiscovery{t}}
	t.Cleanup(func() { http.DefaultClient = oldClient })
	agent := bundledTestAgent(t, "0.1.2")
	tensorfs := installWheel(t, "tensorfs", "0.3.78", "tfs")
	wheel := bundledRuntimeWheel(t, "0.18.87", agent)
	for _, path := range []string{tensorfs, wheel} {
		data, err := os.ReadFile(path)
		must(t, err)
		writeInstallFile(t, filepath.Join(index, filepath.Base(path)), data, 0644)
	}
	h := machines.NewHost(filepath.Join(t.TempDir(), "machine"), "", nil)
	installed, problem := h.Install(context.Background(), machines.Source{}, uv)
	if problem != nil {
		binary := filepath.Join(h.Root(), "opt/cozy/python/bin/cozy-machine")
		info, statErr := os.Stat(binary)
		out, runErr := exec.Command(binary, "version").CombinedOutput()
		t.Fatalf("default bootstrap: %v; module=%s, stat=%v %v, version=%s %v", problem, machines.HostModule(binary), info, statErr, out, runErr)
	}
	if installed.HostPinned || installed.Host.Module != machines.AgentModule || h.Outdated() {
		t.Fatal("default bundled agent was pinned or considered legacy")
	}
	link := filepath.Join(h.Root(), "usr/local/bin/cozy-machine")
	bundle := filepath.Join(h.Root(), "opt/cozy/python/bin/cozy-machine")
	assertLink := func() {
		t.Helper()
		if target, err := os.Readlink(link); err != nil || target != bundle {
			t.Fatalf("agent does not follow Runtime wheel: %q %v", target, err)
		}
	}
	assertLink()
	// A legacy entry-point symlink is replaced without modifying its destination.
	stub := filepath.Join(h.Root(), "usr/local/bin/pod-supervisor")
	if _, err := os.Readlink(stub); err == nil {
		t.Fatal("legacy entry point remains a symlink")
	}
	if code, err := exec.Command(stub).CombinedOutput(); err == nil || !strings.Contains(string(code), "legacy machine control retired") {
		t.Fatalf("legacy entry point did not refuse: %s %v", code, err)
	}
}

func TestMachineAdoptsAvailableBundleWithoutDiscovery(t *testing.T) {
	uv := offlineInstallerUV(t)
	t.Setenv("UV_NO_INDEX", "1")
	agent := bundledTestAgent(t, "0.1.2")
	h := machines.NewHost(filepath.Join(t.TempDir(), "machine"), "", nil)
	_, problem := h.Install(context.Background(), machines.Source{Host: agent,
		RuntimeWheel: bundledRuntimeWheel(t, "0.18.87", agent), TensorFSWheel: installWheel(t, "tensorfs", "0.3.78", "tfs")}, uv)
	fatal(t, problem)
	if !h.Outdated() {
		t.Fatal("unpinned copy still shadows the installed Runtime bundle")
	}
	changed, problem := h.Adopt(context.Background(), true)
	fatal(t, problem)
	if !changed || h.Outdated() {
		t.Fatal("idle adoption did not select the wheel-owned executable")
	}
}

func TestMachineUnbundledBootstrapRefusesWithoutIndependentDiscovery(t *testing.T) {
	uv := offlineInstallerUV(t)
	t.Setenv("UV_NO_INDEX", "1")
	previous := http.DefaultClient
	http.DefaultClient = &http.Client{Transport: noAgentDiscovery{t}}
	t.Cleanup(func() { http.DefaultClient = previous })
	h := machines.NewHost(filepath.Join(t.TempDir(), "machine"), "", nil)
	_, problem := h.Install(context.Background(), machines.Source{RuntimeWheel: installWheel(t, "cozy_runtime", "0.18.86", "cozy-runtime-worker"), TensorFSWheel: installWheel(t, "tensorfs", "0.3.78", "tfs")}, uv)
	if problem == nil || problem.ErrName() != "machine.agent_update_required" {
		t.Fatalf("unbundled wheel accepted: %v", problem)
	}
}

// uv has no UV_NO_INDEX environment option. Use its actual flag so fixtures
// cannot silently substitute a higher-priority native wheel from PyPI.
func offlineInstallerUV(t *testing.T) string {
	t.Helper()
	uv, err := exec.LookPath("uv")
	if err != nil {
		t.Skip("uv is required")
	}
	path := filepath.Join(t.TempDir(), "uv")
	writeInstallFile(t, path, []byte(fmt.Sprintf("#!/bin/sh\nif [ \"$1\" = pip ] && [ \"$2\" = install ]; then shift 2; exec %q pip install --no-index \"$@\"; fi\nexec %q \"$@\"\n", uv, uv)), 0755)
	return path
}

func TestNewMachineInstallRejectsAgentWithoutBootstrapCapability(t *testing.T) {
	agent := bundledTestAgent(t, "99.0.0", machines.HubAccessCapability, machines.RuntimeUpdateCapability)
	for _, explicit := range []bool{true, false} {
		t.Run(fmt.Sprintf("explicit=%v", explicit), func(t *testing.T) {
			dir := t.TempDir()
			host := machines.NewHost(dir, "", nil)
			source := machines.Source{Host: agent, Pinned: true}
			uv := "must-not-run-bootstrap"
			if !explicit {
				source = machines.Source{RuntimeWheel: bundledRuntimeWheel(t, "0.18.87", agent), TensorFSWheel: installWheel(t, "tensorfs", "0.3.78", "tfs")}
				uv = offlineInstallerUV(t)
			}
			_, problem := host.Install(t.Context(), source, uv)
			if problem == nil || problem.ErrName() != "machine.agent_update_required" || !strings.Contains(problem.Message, machines.BootstrapCapability) {
				t.Fatalf("new installation admitted an agent without transactional bootstrap: %v", problem)
			}
			if _, err := os.Stat(filepath.Join(dir, "installed.json")); !os.IsNotExist(err) {
				t.Fatal("refused install recorded successful installation")
			}
			if explicit {
				if _, err := os.Stat(filepath.Join(host.Root(), "opt/cozy/python")); !os.IsNotExist(err) {
					t.Fatal("explicit incompatible agent changed Python before refusal")
				}
			}
		})
	}
}
