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
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"sync/atomic"
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

// machine is a real machine serving on a loopback port, and the key it authorizes.
type machine struct {
	Address, Worker, Cert string
	signer                Signer
}

// serve starts the real machine on a loopback port and answers a machine-scope client.
func serve(t *testing.T, args ...string) *Client {
	m := start(t, args...)
	return m.connect(t, m.Address)
}

func start(t *testing.T, args ...string) machine {
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
	process := exec.Command(*machineBinary, "serve", "--state", state, "--machine-config",
		filepath.Join(config, "machine.json"), "--listen", "127.0.0.1:0", "--host-bytes", "0")
	process.Args = append(process.Args, args...)
	process.Stderr = os.Stderr
	must(t, process.Start())
	t.Cleanup(func() { _ = process.Process.Kill(); _, _ = process.Process.Wait() })
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
	sign := func(message []byte) []byte { return ed25519.Sign(private, message) }
	return machine{Address: ready.Address, Worker: ready.Worker, Cert: ready.Cert, signer: Signer{Public: public, Sign: sign}}
}

// connect dials the machine at addr (its own, or a relay's) as its owner.
func (m machine) connect(t *testing.T, addr string) *Client {
	roots := x509.NewCertPool()
	roots.AppendCertsFromPEM([]byte(m.Cert))
	client, err := Dial(addr, &tls.Config{RootCAs: roots, ServerName: "localhost"}, m.Worker, m.signer)
	must(t, err)
	t.Cleanup(func() { client.Close() })
	return client
}

// relay forwards loopback connections to the machine, counting the bytes it carries there.
// cut, when positive, ends the connection whose bytes first pass it, once.
type relay struct {
	sent atomic.Int64
	cut  int64
	once sync.Once
}

func (r *relay) start(t *testing.T, target string) string {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	must(t, err)
	t.Cleanup(func() { listener.Close() })
	go func() {
		for {
			from, err := listener.Accept()
			if err != nil {
				return
			}
			to, err := net.Dial("tcp", target)
			if err != nil {
				from.Close()
				continue
			}
			go func() { _, _ = io.Copy(from, to); from.Close() }()
			go r.forward(from, to)
		}
	}()
	return listener.Addr().String()
}

func (r *relay) forward(from, to net.Conn) {
	defer to.Close()
	defer from.Close()
	var carried int64
	buffer := make([]byte, 32<<10)
	for {
		n, err := from.Read(buffer)
		if n > 0 {
			if _, err := to.Write(buffer[:n]); err != nil {
				return
			}
			r.sent.Add(int64(n))
			carried += int64(n)
		}
		if err != nil {
			return
		}
		if r.cut > 0 && carried > r.cut {
			ended := false
			r.once.Do(func() { ended = true })
			if ended {
				return
			}
		}
	}
}

// object is length deterministic bytes and their digest.
func object(length int, seed byte) ([]byte, string) {
	data := make([]byte, length)
	for i := range data {
		data[i] = byte(i%251) ^ seed
	}
	sum := sha256.Sum256(data)
	return data, "sha256:" + hex.EncodeToString(sum[:])
}

// heldBy is how many of the object's bytes the machine holds for client's signer.
func heldBy(t *testing.T, client *Client, digest string, length int) uint64 {
	held, err := send(context.Background(), client.Machine, &pb.WriteFrame{Digest: digest, Length: uint64(length)}, nil)
	must(t, err)
	return held
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
	data, digest := object(3*writeChunk+12345, 0)
	// A first attempt ends after one chunk.
	held, err := send(ctx, client.Machine, &pb.WriteFrame{Digest: digest, Length: uint64(len(data))}, bytes.NewReader(data[:writeChunk]))
	must(t, err)
	if held != writeChunk {
		t.Fatalf("held %d after the first chunk", held)
	}
	sent, err := client.write(ctx, digest, uint64(len(data)), bytes.NewReader(data))
	must(t, err)
	again, err := client.write(ctx, digest, uint64(len(data)), bytes.NewReader(data))
	must(t, err)
	if !sent || again {
		t.Fatalf("the rest was sent %t, the held object again %t", sent, again)
	}
	// Bytes that differ from their digest are refused, not stored.
	wrong := "sha256:" + hex.EncodeToString(make([]byte, 32))
	if err := client.Write(ctx, wrong, 3, bytes.NewReader([]byte("abc"))); err == nil {
		t.Fatal("a digest mismatch was accepted")
	}
}

// Runs submitted together write the same package to one machine: its bytes cross once,
// whichever of this process's connections each writer holds.
func TestConcurrentWritesOfOneObjectMoveItsBytesOnce(t *testing.T) {
	m := start(t)
	var r relay
	addr := r.start(t, m.Address)
	clients := []*Client{m.connect(t, addr), m.connect(t, addr)}
	data, digest := object(24<<20, 1)
	var wait sync.WaitGroup
	errs := make(chan error, 4)
	for i := range 4 {
		wait.Go(func() {
			errs <- clients[i%2].Write(context.Background(), digest, uint64(len(data)), bytes.NewReader(data))
		})
	}
	wait.Wait()
	close(errs)
	for err := range errs {
		must(t, err)
	}
	if held := heldBy(t, clients[0], digest, len(data)); held != uint64(len(data)) {
		t.Fatalf("the machine holds %d of %d bytes", held, len(data))
	}
	if sent := r.sent.Load(); sent > int64(len(data))*5/4 {
		t.Fatalf("%d bytes crossed for a %d-byte object written four times at once", sent, len(data))
	}
}

// A connection lost mid-upload costs only what it had in flight: the write resumes on a new one
// from what the machine holds, and the machine ends with the exact object.
func TestAWriteResumesWhenItsConnectionIsCut(t *testing.T) {
	m := start(t)
	data, digest := object(24<<20, 2)
	r := relay{cut: int64(len(data)) / 2}
	client := m.connect(t, r.start(t, m.Address))
	must(t, client.Write(context.Background(), digest, uint64(len(data)), bytes.NewReader(data)))
	if held := heldBy(t, client, digest, len(data)); held != uint64(len(data)) {
		t.Fatalf("the machine holds %d of %d bytes", held, len(data))
	}
	if sent := r.sent.Load(); sent > int64(len(data))*5/4 {
		t.Fatalf("%d bytes crossed for a %d-byte object cut halfway: it started over", sent, len(data))
	}
	// The machine verified the bytes: the stored object is the one named.
	again, err := client.write(context.Background(), digest, uint64(len(data)), bytes.NewReader(data))
	must(t, err)
	if again {
		t.Fatal("the completed object was sent again")
	}
}

// Unpublished code written with LocalSource installs on the real machine inside a warm run.
// Written again unchanged, nothing is offered; after an edit, only the edited file.
func TestLocalSourcePreparesOnTheRealMachine(t *testing.T) {
	if *installerPython == "" || *clientWheel == "" || *cpuFixture == "" {
		t.Skip("requires -installer-python, -client-wheel and -cpu-fixture")
	}
	client := serve(t, "--installer-python", *installerPython, "--client-wheel", *clientWheel)
	ctx := context.Background()
	project := t.TempDir()
	installation := localpackage.Installation{Package: "local/cozy-machine-cpu-lifecycle", Release: "0.1.0"}
	for _, name := range []string{"pyproject.toml", "package.toml", "cpu_lifecycle/__init__.py"} {
		raw, err := os.ReadFile(filepath.Join(*cpuFixture, name))
		must(t, err)
		path := filepath.Join(project, name)
		must(t, os.MkdirAll(filepath.Dir(path), 0o700))
		must(t, os.WriteFile(path, raw, 0o600))
		object, err := fileObject(path)
		must(t, err)
		installation.Files = append(installation.Files, localpackage.File{Name: name, Digest: object.Digest, Length: object.Length, Path: path})
	}
	manifest, sent, held, err := client.LocalSource(ctx, installation, nil, false)
	must(t, err)
	if !sent {
		t.Fatal("new code was not written")
	}
	var stages []string
	spec := &pb.RunSpec{Source: &pb.RunSpec_Local{Local: &pb.LocalSource{Manifest: manifest}}, Entrypoint: "steps", Owner: "alice"}
	must(t, Prepare(ctx, client.Machine, "warm-1", spec, func(p Progress) { stages = append(stages, p.Stage) }))
	if len(stages) == 0 {
		t.Fatal("the warm run reported no progress")
	}
	again, sent, _, err := client.LocalSource(ctx, installation, held, false)
	must(t, err)
	if again != manifest || sent {
		t.Fatalf("the same code named another manifest (%s, then %s) or was written again (%t)", manifest, again, sent)
	}
	// The edit: one file's bytes move, and the machine installs it in the tree it holds.
	edited := &installation.Files[len(installation.Files)-1]
	raw, err := os.ReadFile(edited.Path)
	must(t, err)
	must(t, os.WriteFile(edited.Path, append(raw, []byte("\nEDITED = True\n")...), 0o600))
	object, err := fileObject(edited.Path)
	must(t, err)
	edited.Digest, edited.Length = object.Digest, object.Length
	next, sent, _, err := client.LocalSource(ctx, installation, held, false)
	must(t, err)
	if next == manifest || !sent {
		t.Fatalf("the edit named the same manifest or moved no byte (%s, %t)", next, sent)
	}
	spec.Source = &pb.RunSpec_Local{Local: &pb.LocalSource{Manifest: next}}
	must(t, Prepare(ctx, client.Machine, "warm-2", spec, nil))
	// This computer's machine reads the project where it is: no file of it is written.
	installation.Root = project
	inPlace, _, kept, err := client.LocalSource(ctx, installation, nil, true)
	must(t, err)
	if len(kept) != len(installation.Files) {
		t.Fatalf("in place, the machine holds %d of %d files by name", len(kept), len(installation.Files))
	}
	spec.Source = &pb.RunSpec_Local{Local: &pb.LocalSource{Manifest: inPlace}}
	must(t, Prepare(ctx, client.Machine, "warm-3", spec, nil))
}
