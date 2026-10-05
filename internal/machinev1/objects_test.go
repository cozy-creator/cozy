package machinev1

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"flag"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/localpackage"
	pb "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

var (
	machineBinary   = flag.String("machine-binary", "", "a cozy-machine binary to serve these tests")
	installerPython = flag.String("installer-python", "", "the machine's package installer Python (cozy_machine_client importable)")
	clientWheel     = flag.String("client-wheel", "", "the machine's CPU runner client wheel")
	cpuFixture      = flag.String("cpu-fixture", "", "cozy-machine tests/fixtures/cpu_lifecycle")
)

// serve starts the real machine on a loopback port and answers a machine-scope client.
func serve(t *testing.T, args ...string) pb.MachineClient {
	if *machineBinary == "" {
		t.Skip("requires -machine-binary=<cozy-machine>")
	}
	root := t.TempDir()
	public, private, _ := ed25519.GenerateKey(nil)
	config := filepath.Join(root, "config")
	must(t, os.MkdirAll(config, 0o700))
	put := func(name string, value any) {
		data, _ := json.Marshal(value)
		must(t, os.WriteFile(filepath.Join(config, name), data, 0o600))
	}
	put("keys.json", map[string]any{"keys": []string{base64.RawURLEncoding.EncodeToString(public)}})
	put("readiness.json", map[string]any{"key_b64url": base64.RawURLEncoding.EncodeToString(bytes.Repeat([]byte{7}, 32))})
	put("machine.json", map[string]any{"worker_id": "cli-objects", "identity_directory": "identity",
		"authorized_keys_file": "keys.json", "readiness_hmac_key_file": "readiness.json"})
	state := filepath.Join(root, "state")
	machine := exec.Command(*machineBinary, "serve", "--state", state, "--machine-config",
		filepath.Join(config, "machine.json"), "--listen", "127.0.0.1:0", "--host-bytes", "0")
	machine.Args = append(machine.Args, args...)
	machine.Stderr = os.Stderr
	must(t, machine.Start())
	t.Cleanup(func() { _ = machine.Process.Kill(); _, _ = machine.Process.Wait() })
	var ready struct {
		Address string `json:"address"`
		Worker  string `json:"worker_id"`
		Cert    string `json:"cert_pem"`
	}
	// Test harness bound on a broken start; the product has no such limit.
	for until := time.Now().Add(2 * time.Minute); ; time.Sleep(50 * time.Millisecond) {
		if data, err := os.ReadFile(filepath.Join(state, "api-ready.json")); err == nil && json.Unmarshal(data, &ready) == nil {
			break
		}
		if time.Now().After(until) {
			t.Fatal("the machine did not publish api-ready.json")
		}
	}
	roots := x509.NewCertPool()
	roots.AppendCertsFromPEM([]byte(ready.Cert))
	sign := func(message []byte) []byte { return ed25519.Sign(private, message) }
	client, err := Dial(ready.Address, &tls.Config{RootCAs: roots, ServerName: "localhost"}, ready.Worker, Signer{Public: public, Sign: sign})
	must(t, err)
	t.Cleanup(func() { client.Close() })
	return client.Machine
}

func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

// An interrupted upload resumes from what the machine holds, and a written object is never
// sent again.
func TestWriteResumesFromHeldBytesOnTheRealMachine(t *testing.T) {
	client := serve(t)
	ctx := context.Background()
	data := make([]byte, 3*writeChunk+12345)
	for i := range data {
		data[i] = byte(i % 251)
	}
	sum := sha256.Sum256(data)
	digest := "sha256:" + hex.EncodeToString(sum[:])
	length := int64(len(data))
	// A first attempt ends after one chunk.
	held, err := write(ctx, client, &pb.WriteFrame{Digest: digest, Length: uint64(length)}, bytes.NewReader(data[:writeChunk]))
	must(t, err)
	if held != writeChunk {
		t.Fatalf("held %d after the first chunk", held)
	}
	sent := 0
	open := func() (io.ReadSeekCloser, error) {
		sent++
		return nopCloser{bytes.NewReader(data)}, nil
	}
	must(t, Write(ctx, client, digest, length, open))
	must(t, Write(ctx, client, digest, length, open))
	if sent != 1 {
		t.Fatalf("the bytes were opened %d times; a held object needs none", sent)
	}
	// Bytes that differ from their digest are refused, not stored.
	wrong := "sha256:" + hex.EncodeToString(make([]byte, 32))
	if err := Write(ctx, client, wrong, 3, func() (io.ReadSeekCloser, error) { return nopCloser{bytes.NewReader([]byte("abc"))}, nil }); err == nil {
		t.Fatal("a digest mismatch was accepted")
	}
}

// Unpublished code written with LocalSource installs on the real machine inside a warm run,
// and the same code written again reuses every held object.
func TestLocalSourcePreparesOnTheRealMachine(t *testing.T) {
	if *installerPython == "" || *clientWheel == "" || *cpuFixture == "" {
		t.Skip("requires -installer-python, -client-wheel and -cpu-fixture")
	}
	client := serve(t, "--installer-python", *installerPython, "--client-wheel", *clientWheel)
	ctx := context.Background()
	archive := filepath.Join(t.TempDir(), "source.tar")
	tar := exec.Command("tar", "-cf", archive, "-C", *cpuFixture, "pyproject.toml", "package.toml", "cpu_lifecycle/__init__.py")
	if output, err := tar.CombinedOutput(); err != nil {
		t.Fatalf("%v: %s", err, output)
	}
	installation := localpackage.Installation{Package: "local/cozy-machine-cpu-lifecycle", Release: "0.1.0", PythonVersion: "3.12",
		Files: []localpackage.File{{Kind: "source", Filename: "source.tar", Path: archive}}}
	manifest, sent, err := LocalSource(ctx, client, installation)
	must(t, err)
	if !sent {
		t.Fatal("new code was not written")
	}
	var stages []string
	spec := &pb.RunSpec{Source: &pb.RunSpec_Local{Local: &pb.LocalSource{Manifest: manifest}}, Entrypoint: "steps", Owner: "alice"}
	must(t, Prepare(ctx, client, "warm-1", spec, func(p Progress) { stages = append(stages, p.Stage) }))
	if len(stages) == 0 {
		t.Fatal("the warm run reported no progress")
	}
	again, sent, err := LocalSource(ctx, client, installation)
	must(t, err)
	if again != manifest || sent {
		t.Fatalf("the same code named another manifest (%s, then %s) or was written again (%t)", manifest, again, sent)
	}
}
