package producttest

import (
	"archive/zip"
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math/big"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/machines"
)

var (
	machineHostBinary    = flag.String("machine-host", "", "the standalone cozy-machine executable each test machine runs")
	requireMachineHost   = flag.Bool("require-machine-host", false, "fail, never skip, a local execution the run cannot host (CI)")
	machineRuntimePython = flag.String("machine-runtime-python", "", "Development interpreter for the standalone agent public-view media fixture")
	machineRuntimeWheel  = flag.String("machine-runtime-wheel", "", "Runtime wheel the test machines run; default: the published Runtime")
	machineUpdateWheel   = flag.String("machine-update-wheel", "", "Candidate Runtime wheel with a distinct bundled agent for local update proof")
	machineTensorFSWheel = flag.String("machine-tensorfs-wheel", "", "TensorFS wheel paired with -machine-runtime-wheel")
)

// One installed machine layout for the whole run; each root's machine links its executables,
// as a pod's image is shared and its / is its own.
var machineTemplate struct {
	once    sync.Once
	dir     string
	problem *exit.Error
}

func machineTemplateDir(t testing.TB) string {
	t.Helper()
	machineTemplate.once.Do(func() {
		machineTemplate.dir = filepath.Join(scratchBase, "machine-template")
		uv, err := exec.LookPath("uv")
		if err != nil {
			machineTemplate.problem = exit.New(exit.NotFound, "uv lays out the test machines: %s", err)
			return
		}
		source := machines.Source{Host: *machineHostBinary, RuntimeWheel: *machineRuntimeWheel, TensorFSWheel: *machineTensorFSWheel}
		_, machineTemplate.problem = machines.NewHost(machineTemplate.dir, "", nil).Install(context.Background(), source, uv)
	})
	if machineTemplate.problem != nil {
		t.Fatal(machineTemplate.problem)
	}
	return machineTemplate.dir
}

// provisionMachine gives a test root an independent machine when the run names an agent.
// Its local identity needs no Hub registration; each request delegates only the Hub access
// its execution requires.
func provisionMachine(t *testing.T, root string) {
	t.Helper()
	provisionMachineIn(t, root, "")
}

// unpressuredMachine gives root its machine on a filesystem the machine's own Runtime finds
// free of storage pressure. Under pressure the Runtime evicts every unused memo entry, as it
// should; a test of memo reuse needs the headroom a user's machine would have.
func unpressuredMachine(t *testing.T, root string) {
	t.Helper()
	if *machineHostBinary == "" {
		return
	}
	python := machinePython(t)
	for _, parent := range []string{os.TempDir(), "/dev/shm"} {
		out, err := exec.Command(python, "-I", "-c", `import sys
from pathlib import Path
from cozy_runtime.internal.local_storage_admission import pressure_target
print(pressure_target(Path(sys.argv[1])))`, parent).Output()
		if err == nil && strings.TrimSpace(string(out)) == "0" {
			provisionMachineIn(t, root, parent)
			return
		}
	}
	t.Fatal("every filesystem for the machine is under storage pressure; its Runtime would evict the memo entries this test reuses")
}

// machinePython is an interpreter holding the SDK pair the test machines run (the Runtime and
// TensorFS wheels of the machine template), made once per run. A machine root has no
// interpreter of its own: fixtures that seed a store or ask the Runtime a fact use this one.
func machinePython(t testing.TB) string {
	t.Helper()
	machineSDK.once.Do(func() {
		wheels, _ := filepath.Glob(filepath.Join(machineTemplateDir(t), "root/opt/cozy/machine/wheels/*.whl"))
		venv := filepath.Join(scratchBase, "machine-python")
		python := filepath.Join(venv, "bin", "python")
		for _, args := range [][]string{{"venv", "--python", "3.12", venv}, append([]string{"pip", "install", "--python", python}, wheels...)} {
			if out, err := exec.Command("uv", args...).CombinedOutput(); err != nil {
				machineSDK.err = fmt.Errorf("uv %s for the machine SDK interpreter: %v\n%s", args[0], err, out)
				return
			}
		}
		machineSDK.python = python
	})
	if machineSDK.err != nil {
		t.Fatal(machineSDK.err)
	}
	return machineSDK.python
}

var machineSDK struct {
	once   sync.Once
	python string
	err    error
}

func provisionMachineIn(t *testing.T, root, parent string) {
	t.Helper()
	if *machineHostBinary == "" {
		return
	}
	dir := filepath.Join(root, "machine")
	if _, err := os.Lstat(dir); err == nil {
		return
	}
	template := machineTemplateDir(t)
	// The machine lives at a short path, as a pod's does at /: a test root's own path would
	// push the Runtime's executor sockets past the kernel's socket path bound.
	short, err := os.MkdirTemp(parent, "cm")
	must(t, err)
	must(t, os.MkdirAll(root, 0o700))
	must(t, os.Symlink(short, dir))
	t.Cleanup(func() {
		reapMachineRuntimeRoot(root)
		if !t.Failed() {
			_ = removeAllForce(short)
		}
	})
	// Everything `cozy machine install` lays out: the machine, its uv and the executor SDK
	// wheels package environments install from.
	for _, link := range []string{"usr/local/bin/cozy-machine", "usr/local/bin/uv", "opt/cozy/machine/wheels"} {
		target, err := filepath.EvalSymlinks(filepath.Join(template, "root", link))
		must(t, err)
		path := filepath.Join(dir, "root", link)
		must(t, os.MkdirAll(filepath.Dir(path), 0o755))
		if strings.HasSuffix(link, "cozy-machine") {
			// Its own path, so the running Host is found and stopped by this root's teardown.
			if os.Link(target, path) != nil {
				raw, err := os.ReadFile(target)
				must(t, err)
				must(t, os.WriteFile(path, raw, 0o755))
			}
			continue
		}
		must(t, os.Symlink(target, path))
	}
	installed, err := os.ReadFile(filepath.Join(template, "installed.json"))
	must(t, err)
	must(t, os.WriteFile(filepath.Join(dir, "installed.json"), installed, 0o600))
}

// hubTLSServer serves handler as a Hub with a private CA does: the CA signs a 127.0.0.1 leaf
// and a grant names the CA's DER. uv, like any WebPKI client, refuses a CA as a leaf.
func hubTLSServer(t *testing.T, handler http.Handler) (*httptest.Server, []byte) {
	t.Helper()
	certificate := func(template, parent *x509.Certificate, key *ecdsa.PrivateKey, signer *ecdsa.PrivateKey) []byte {
		template.NotBefore, template.NotAfter = time.Now().Add(-time.Hour), time.Now().Add(24*time.Hour)
		der, err := x509.CreateCertificate(rand.Reader, template, parent, &key.PublicKey, signer)
		must(t, err)
		return der
	}
	caKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	must(t, err)
	caTemplate := &x509.Certificate{SerialNumber: big.NewInt(1), Subject: pkix.Name{CommonName: "test Hub CA"},
		IsCA: true, BasicConstraintsValid: true, KeyUsage: x509.KeyUsageCertSign}
	caDER := certificate(caTemplate, caTemplate, caKey, caKey)
	ca, err := x509.ParseCertificate(caDER)
	must(t, err)
	leafKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	must(t, err)
	leaf := certificate(&x509.Certificate{SerialNumber: big.NewInt(2), Subject: pkix.Name{CommonName: "127.0.0.1"},
		IPAddresses: []net.IP{net.IPv4(127, 0, 0, 1)}, KeyUsage: x509.KeyUsageDigitalSignature,
		ExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth}}, ca, leafKey, caKey)
	server := httptest.NewUnstartedServer(handler)
	server.TLS = &tls.Config{Certificates: []tls.Certificate{{Certificate: [][]byte{leaf}, PrivateKey: leafKey}}}
	server.StartTLS()
	return server, caDER
}

// skipWithoutMachine turns a local execution this run cannot host into a skip: the suite
// gives each root a machine only when it is told which Host binary to run. A run that
// requires the Host (CI) fails instead.
func skipWithoutMachine(t *testing.T, code int, output string) {
	t.Helper()
	if code == 0 || *machineHostBinary != "" || !strings.Contains(output, "machine.not_installed") {
		return
	}
	if *requireMachineHost {
		t.Fatalf("local execution needs this computer's machine and the run has no -machine-host:\n%s", output)
	}
	t.Skip("local execution runs on this computer's machine; pass -machine-host=<standalone cozy-machine>")
}

// A run that requires the machine Host proves it has one before any test relies on it.
func TestMachineHostIsPresentWhenRequired(t *testing.T) {
	if !*requireMachineHost {
		t.Skip("the run does not require the machine Host")
	}
	if *machineHostBinary == "" {
		t.Fatal("-require-machine-host without -machine-host: local execution tests cannot run")
	}
	if info, err := os.Stat(*machineHostBinary); err != nil || info.Mode()&0o111 == 0 {
		t.Fatalf("-machine-host %s is not an executable: %v", *machineHostBinary, err)
	}
	machineTemplateDir(t)
}

// stubMachine gives root an installed current machine whose agent is script: a sentinel
// that proves a launch was attempted without running one.
func stubMachine(t *testing.T, root, script string) {
	t.Helper()
	dir := filepath.Join(root, "machine")
	binary := filepath.Join(dir, "root", "usr", "local", "bin", "cozy-machine")
	must(t, os.MkdirAll(filepath.Dir(binary), 0o755))
	must(t, os.WriteFile(binary, []byte(script), 0o700)) //cozy:allow sentinel Host proves whether a launch was attempted
	metadata := `{"host":{"name":"cozy-machine"},"host_pinned":true}`
	must(t, os.WriteFile(filepath.Join(dir, "installed.json"), []byte(metadata), 0600))

}

// machineStore matches the isolated TENSORFS_HOME supplied by childEnv to every
// ordinary CLI invocation in these product fixtures.
func machineStore(root string) string {
	return filepath.Join(root, "tensorfs")
}

// storeReader is a Python that opens the machine's store. The plane-only TensorFS a machine's
// executors install has no store, so this takes the full build from the package index.
func storeReader(t *testing.T) string {
	t.Helper()
	env := filepath.Join(t.TempDir(), "store-reader")
	python := filepath.Join(env, "bin", "python")
	for _, args := range [][]string{{"venv", env, "--python", "3.12"}, {"pip", "install", "--python", python, "tensorfs"}} {
		if out, err := exec.Command("uv", args...).CombinedOutput(); err != nil {
			t.Fatalf("uv %s: %v\n%s", args[0], err, out)
		}
	}
	return python
}

// machineInstallations holds the package environments this computer's machine prepared: one
// generation each, its packages in `env` and an unpublished root's code in `source`.
func machineInstallations(root string) string {
	return filepath.Join(root, "machine", "root", "var", "lib", "cozy", "rust-machine", "generations")
}

// machineJournal is this computer's machine's execution journal: the Runtime's workspace in
// the machine's own TensorFS Store.
func machineJournal(root string) string {
	return filepath.Join(machineStore(root), ".cozy-workspace", "journal.sqlite3")
}

func transportLogs(root string) string {
	paths, _ := filepath.Glob(filepath.Join(root, "tmp", "runtime-updates", "*", "transport.log"))
	text := ""
	for _, path := range paths {
		data, _ := os.ReadFile(path)
		text += path + ":\n" + string(data) + "\n"
	}
	return text
}

// eventually waits for ok; the bound only catches a hang on a loaded shared box.
func eventually(t *testing.T, root, what string, ok func() bool) {
	t.Helper()
	for deadline := time.Now().Add(5 * time.Minute); !ok(); time.Sleep(50 * time.Millisecond) {
		if time.Now().After(deadline) {
			t.Fatalf("%s did not happen: %s\n%s", what, tail(filepath.Join(root, "daemon.log")), transportLogs(root))
		}
	}
}

// cozyWithin runs one CLI command, failing the test if it outlives `within`.
func cozyWithin(t *testing.T, root string, within time.Duration, args ...string) (int, string) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), within)
	defer cancel()
	cmd := exec.CommandContext(ctx, "/usr/bin/nice", append([]string{"-n", "19", cozyBin}, args...)...)
	cmd.Env = childEnv(t, root)
	data, _ := cmd.CombinedOutput()
	if ctx.Err() != nil {
		t.Fatalf("cozy %s hung past %s: %s\n%s", strings.Join(args, " "), within, data, tail(filepath.Join(root, "daemon.log")))
	}
	return cmd.ProcessState.ExitCode(), string(data)
}

// runtimeFloor is the Runtime the test packages' declared dependencies admit.
const runtimeFloor = "0.18.85"

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

// The isolated Hub fixture supplies a syntactic delegated JWT. The real Hub verifies its
// authority; the agent only needs its stable issuer/account identity to guard replacement.
func executionGrantToken(issuer, principal string, serial int32) string {
	header := base64.RawURLEncoding.EncodeToString([]byte(`{"typ":"delegated-access+jwt"}`))
	body, _ := json.Marshal(map[string]any{"iss": issuer, "delegated_sub": principal, "permissions": []string{"cozy.execution-access"}, "jti": serial, "attributes": map[string]any{"execution_device_key_id": "fixture-device"}})
	return header + "." + base64.RawURLEncoding.EncodeToString(body) + "." + base64.RawURLEncoding.EncodeToString([]byte("fixture-signature"))
}

func installWheel(t *testing.T, name, version, entrypoint string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name+"-"+version+"-py3-none-any.whl")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	w := zip.NewWriter(f)
	dist := name + "-" + version + ".dist-info/"
	files := map[string]string{
		name + ".py":      "def main(): pass\n",
		dist + "METADATA": "Metadata-Version: 2.1\nName: " + strings.ReplaceAll(name, "_", "-") + "\nVersion: " + version + "\nProvides-Extra: media\n",
		dist + "WHEEL":    "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
	}
	if name == "cozy_runtime" {
		files[dist+"METADATA"] += "Requires-Dist: tensorfs>=0.3.78\n"
	}
	if entrypoint != "" {
		files[dist+"entry_points.txt"] = "[console_scripts]\n" + entrypoint + " = " + name + ":main\n"
	}
	var record strings.Builder
	for path := range files {
		fmt.Fprintln(&record, path+",,")
	}
	files[dist+"RECORD"] = record.String() + dist + "RECORD,,\n"
	for path, body := range files {
		file, err := w.Create(path)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := file.Write([]byte(body)); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	return path
}

func sha256Hex(raw []byte) string {
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:])
}
