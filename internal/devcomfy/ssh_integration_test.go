package devcomfy

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"golang.org/x/crypto/ssh"
	"golang.org/x/crypto/ssh/knownhosts"
)

// Real OpenSSH authentication + host pin + remote Python + HTTP + retained files.
// Only the GPU service itself is a CPU fixture; no Python package executor runs.
func TestOwnedSSHComfyRoundTripAndReplay(t *testing.T) {
	for _, name := range []string{"ssh", "ssh-keygen", "python3"} {
		if _, err := exec.LookPath(name); err != nil {
			t.Skip(name, "not installed")
		}
	}
	root := t.TempDir()
	key := filepath.Join(root, "key")
	if raw, err := exec.Command("ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", key).CombinedOutput(); err != nil {
		t.Fatal(err, string(raw))
	}
	publicRaw, _ := os.ReadFile(key + ".pub")
	public, _, _, _, err := ssh.ParseAuthorizedKey(publicRaw)
	if err != nil {
		t.Fatal(err)
	}
	_, hostPrivate, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	signer, err := ssh.NewSignerFromKey(hostPrivate)
	if err != nil {
		t.Fatal(err)
	}
	config := &ssh.ServerConfig{PublicKeyCallback: func(meta ssh.ConnMetadata, k ssh.PublicKey) (*ssh.Permissions, error) {
		if meta.User() != "root" || !bytes.Equal(k.Marshal(), public.Marshal()) {
			return nil, fmt.Errorf("wrong owner")
		}
		return &ssh.Permissions{}, nil
	}}
	config.AddHostKey(signer)
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	var sessions sync.WaitGroup
	go func() {
		for {
			conn, err := listener.Accept()
			if err != nil {
				return
			}
			sessions.Add(1)
			go func() {
				defer sessions.Done()
				defer conn.Close()
				server, channels, requests, err := ssh.NewServerConn(conn, config)
				if err != nil {
					return
				}
				defer server.Close()
				go ssh.DiscardRequests(requests)
				for pending := range channels {
					if pending.ChannelType() != "session" {
						pending.Reject(ssh.UnknownChannelType, "session only")
						continue
					}
					channel, requests, err := pending.Accept()
					if err != nil {
						continue
					}
					for request := range requests {
						if request.Type != "exec" {
							request.Reply(false, nil)
							continue
						}
						var spec struct{ Command string }
						if ssh.Unmarshal(request.Payload, &spec) != nil {
							request.Reply(false, nil)
							break
						}
						request.Reply(true, nil)
						command := exec.Command("sh", "-c", spec.Command)
						command.Env = append(os.Environ(), "CUDA_VISIBLE_DEVICES=")
						command.Stdin = channel
						command.Stdout = channel
						command.Stderr = channel.Stderr()
						code := 0
						if err := command.Run(); err != nil {
							code = 1
						}
						channel.SendRequest("exit-status", false, ssh.Marshal(struct{ Status uint32 }{uint32(code)}))
						channel.Close()
						break
					}
				}
			}()
		}
	}()
	defer func() { listener.Close(); sessions.Wait() }()
	host, port, _ := net.SplitHostPort(listener.Addr().String())
	hosts := filepath.Join(root, "known hosts")
	if err := os.WriteFile(hosts, []byte(knownhosts.Line([]string{net.JoinHostPort(host, port)}, signer.PublicKey())+"\n"), 0600); err != nil {
		t.Fatal(err)
	}
	outputs := filepath.Join(root, "remote-output")
	os.Mkdir(outputs, 0700)
	original := bytes.Repeat([]byte("original"), 4096)
	os.WriteFile(filepath.Join(outputs, "sample.mp4"), original, 0600)
	var posts atomic.Int32
	var prompt string
	var mu sync.Mutex
	comfy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.URL.Path == "/queue":
			json.NewEncoder(w).Encode(map[string]any{"queue_running": []any{}, "queue_pending": []any{}})
		case r.URL.Path == "/prompt":
			var body map[string]any
			json.NewDecoder(r.Body).Decode(&body)
			posts.Add(1)
			prompt = body["prompt_id"].(string)
			json.NewEncoder(w).Encode(map[string]any{"prompt_id": prompt})
		case strings.HasPrefix(r.URL.Path, "/history/"):
			json.NewEncoder(w).Encode(map[string]any{prompt: map[string]any{"status": map[string]any{"status_str": "success", "messages": []any{[]any{"execution_start", map[string]any{"timestamp": 1000}}, []any{"execution_success", map[string]any{"timestamp": 2500}}}}, "outputs": map[string]any{"1": map[string]any{"videos": []any{map[string]any{"type": "output", "filename": "sample.mp4"}}}}}})
		default:
			w.WriteHeader(404)
		}
	}))
	defer comfy.Close()
	_, httpPort, _ := net.SplitHostPort(strings.TrimPrefix(comfy.URL, "http://"))
	servicePort, _ := strconv.Atoi(httpPort)
	var renewals atomic.Int32
	o := Options{SSH: SSH{Host: host, Port: port, Key: key, KnownHosts: hosts, Python: "python3", Environment: os.Environ()}, Input: Input{GraphJSON: `{"1":{"class_type":"Fixture","inputs":{"text":"$(touch no)"}}}`, OutputRoot: outputs, Port: servicePort, TimeoutS: 30, ExpectedSteps: 8}, Rental: "rental", Worker: "worker", Boot: "boot", Key: "stable", StateRoot: filepath.Join(root, "remote-state"), ReceiptDir: filepath.Join(root, "receipts"), OutputDir: filepath.Join(root, "local-output"), Keepalive: func(context.Context) error { renewals.Add(1); return nil }}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	first, err := Run(ctx, o)
	if err != nil {
		t.Fatal(err)
	}
	if first.Status != "success" {
		t.Fatalf("%+v", first)
	}
	got, err := os.ReadFile(filepath.Join(o.OutputDir, "outputs/sample.mp4"))
	if err != nil || !bytes.Equal(got, original) {
		t.Fatal("original artifact changed", err)
	}
	second, err := Run(ctx, o)
	if err != nil {
		t.Fatal(err)
	}
	if first.PromptID != second.PromptID || posts.Load() != 1 || renewals.Load() == 0 {
		t.Fatal(first, second, posts.Load(), renewals.Load())
	}
	// A detached download resumes at its retained prefix and verifies the whole hash.
	target := filepath.Join(o.OutputDir, "outputs/sample.mp4")
	if err = os.Remove(target); err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(target+".partial-"+Operation(o.Rental, o.Key)[:12], original[:101], 0600); err != nil {
		t.Fatal(err)
	}
	if _, err = Run(ctx, o); err != nil {
		t.Fatal(err)
	}
	got, err = os.ReadFile(target)
	if err != nil || !bytes.Equal(got, original) || posts.Load() != 1 {
		t.Fatal("partial transfer did not resume the same original", err)
	}
	changed := o
	changed.Boot = "different"
	if _, err = Run(ctx, changed); err == nil {
		t.Fatal("changed boot accepted")
	}
	bad := o
	bad.ReceiptDir = filepath.Join(root, "bad-pin")
	bad.Key = "new-key"
	bad.SSH.KnownHosts = filepath.Join(root, "empty-hosts")
	os.WriteFile(bad.SSH.KnownHosts, nil, 0600)
	started := time.Now()
	if _, err = Run(ctx, bad); err == nil || time.Since(started) > 5*time.Second {
		t.Fatal("host-key refusal was retried or accepted", err)
	}
	if posts.Load() != 1 {
		t.Fatal("bad host pin submitted a graph")
	}
}
