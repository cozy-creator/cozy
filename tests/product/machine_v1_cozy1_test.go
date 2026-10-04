package producttest

import (
	"bytes"
	"cmp"
	"context"
	"crypto/ed25519"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"image"
	"image/png"
	"net"
	"net/netip"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/capability"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/tests/product/webrtctest"
)

// cozy1Root is a home whose machine is granted a WebRTC port, as the Hub grants a rental's.
func cozy1Root(t *testing.T) (root string, port int) {
	t.Helper()
	root, err := os.MkdirTemp(os.TempDir(), "czw")
	must(t, err)
	probe, err := net.Listen("tcp", "127.0.0.1:0") //cozy:allow choosing a free port for the grant
	must(t, err)
	port = probe.Addr().(*net.TCPAddr).Port
	must(t, probe.Close())
	must(t, os.WriteFile(filepath.Join(root, config.FileName), fmt.Appendf(nil, "machine:\n  webrtc_port: %d\n", port), 0o600))
	provisionMachine(t, root)
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "machine", "stop")
		_, _ = runCozy(t, root, "down")
		if t.Failed() {
			t.Logf("evidence retained at %s\nmachine log tail:\n%s", root, tail(filepath.Join(root, "machine", "host.log")))
		} else {
			_ = removeAllForce(root)
		}
	})
	return root, port
}

// cozy1Run starts `cozy run` with --await and answers once its machine runs it: the run's id,
// and the command to wait for.
func cozy1Run(t *testing.T, root string, args ...string) (id string, command *exec.Cmd, ran *bytes.Buffer) {
	t.Helper()
	command = exec.Command("/usr/bin/nice", append([]string{"-n", "19", cozyBin, "run"}, args...)...)
	command.Env = childEnv(t, root)
	ran = &bytes.Buffer{}
	command.Stdout, command.Stderr = ran, ran
	must(t, command.Start())
	// A test that fails before it waits for the command never leaves it running.
	t.Cleanup(func() {
		if command.ProcessState == nil {
			_ = command.Process.Kill()
			_ = command.Wait()
		}
	})
	var run struct {
		ID     string `json:"request_id"`
		Status string `json:"status"`
	}
	landed(t, "the machine to accept the run", func() bool {
		_, shown := runCozy(t, root, "run", "show", "1", "--json")
		return json.Unmarshal([]byte(lastJSONLine(shown)), &run) == nil && run.Status == "in_progress"
	})
	return run.ID, command, ran
}

// cozy1Sessions opens welcomed cozy/1 sessions on root's machine as a browser holding a link
// does: the port its receipt names, its leaf's fingerprint, and a capability for run that an
// authorized key signs.
func cozy1Sessions(t *testing.T, root string, port int, run string) (*mediaHarness, func(capability.Grant) *webrtctest.Client) {
	t.Helper()
	dir := filepath.Join(root, "machine")
	var envelope struct {
		Payload []byte `json:"payload"`
	}
	var receipt struct {
		WebRTC *struct {
			Port int `json:"port"`
		} `json:"webrtc"`
	}
	raw, err := os.ReadFile(filepath.Join(dir, "root/run/cozy/bootstrap/readiness-envelope.json"))
	must(t, err)
	must(t, json.Unmarshal(raw, &envelope))
	must(t, json.Unmarshal(envelope.Payload, &receipt))
	if receipt.WebRTC == nil || receipt.WebRTC.Port != port {
		t.Fatalf("a machine granted WebRTC port %d names %+v in its receipt", port, receipt.WebRTC)
	}
	host := machines.NewHost(dir, "", nil)
	state, problem := host.Status()
	fatal(t, problem)
	owner, problem := host.Owner()
	fatal(t, problem)
	public, err := base64.RawURLEncoding.DecodeString(owner.PublicKey())
	must(t, err)
	raw, err = os.ReadFile(filepath.Join(dir, "leaf.pem"))
	must(t, err)
	leaf, _ := pem.Decode(raw)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	t.Cleanup(cancel)
	h := &mediaHarness{t: t, ctx: ctx}
	return h, func(g capability.Grant) *webrtctest.Client {
		g.Machine, g.Run, g.Expires = state.MachineID, run, time.Now().Add(10*time.Minute).Unix()
		token, err := capability.MintSigned(ed25519.PublicKey(public), owner.Sign, g)
		must(t, err)
		c, err := webrtctest.Dial(ctx, netip.AddrPortFrom(netip.MustParseAddr("127.0.0.1"), uint16(port)),
			webrtctest.Fingerprint(leaf.Bytes), webrtctest.Options{})
		must(t, err)
		t.Cleanup(func() { c.Close() })
		h.send(c, map[string]any{"t": "hello", "v": 1, "cap": token})
		h.expect(c, "welcome")
		h.send(c, map[string]any{"t": "credit", "bytes": 1 << 30})
		return c
	}
}

// cozy1Machine is root's running Rust machine as a browser holding a link reaches it: the
// WebRTC port its receipt names, its leaf's fingerprint, and the owner key that signs caps.
type cozy1Machine struct {
	t           *testing.T
	root        string
	Addr        netip.AddrPort
	Fingerprint string
	Machine     string
	public      ed25519.PublicKey
	sign        func([]byte) []byte
}

func newCozy1Machine(t *testing.T, root string, port int) *cozy1Machine {
	t.Helper()
	dir := filepath.Join(root, "machine")
	var envelope struct {
		Payload []byte `json:"payload"`
	}
	var receipt struct {
		WebRTC *struct {
			Port int `json:"port"`
		} `json:"webrtc"`
	}
	raw, err := os.ReadFile(filepath.Join(dir, "root/run/cozy/bootstrap/readiness-envelope.json"))
	must(t, err)
	must(t, json.Unmarshal(raw, &envelope))
	must(t, json.Unmarshal(envelope.Payload, &receipt))
	if receipt.WebRTC == nil || receipt.WebRTC.Port != port {
		t.Fatalf("a machine granted WebRTC port %d names %+v in its receipt", port, receipt.WebRTC)
	}
	host := machines.NewHost(dir, "", nil)
	state, problem := host.Status()
	fatal(t, problem)
	owner, problem := host.Owner()
	fatal(t, problem)
	public, err := base64.RawURLEncoding.DecodeString(owner.PublicKey())
	must(t, err)
	raw, err = os.ReadFile(filepath.Join(dir, "leaf.pem"))
	must(t, err)
	leaf, _ := pem.Decode(raw)
	conn, err := net.DialTimeout("tcp", net.JoinHostPort("127.0.0.1", fmt.Sprint(port)), 2*time.Second)
	must(t, err)
	must(t, conn.Close())
	return &cozy1Machine{t: t, root: root, Addr: netip.AddrPortFrom(netip.MustParseAddr("127.0.0.1"), uint16(port)),
		Fingerprint: webrtctest.Fingerprint(leaf.Bytes), Machine: state.MachineID, public: public, sign: owner.Sign}
}

// ownerKey is the key this computer's machine authorizes.
func (m *cozy1Machine) ownerKey() ed25519.PrivateKey {
	raw, err := os.ReadFile(filepath.Join(m.root, "machine", "owner.pem"))
	must(m.t, err)
	block, _ := pem.Decode(raw)
	parsed, err := x509.ParsePKCS8PrivateKey(block.Bytes)
	must(m.t, err)
	return parsed.(ed25519.PrivateKey)
}

// mint signs g with the owner key, for this machine and for ten minutes unless g says otherwise.
func (m *cozy1Machine) mint(g capability.Grant) string {
	g.Machine = cmp.Or(g.Machine, m.Machine)
	if g.Expires == 0 {
		g.Expires = time.Now().Add(10 * time.Minute).Unix()
	}
	token, err := capability.MintSigned(m.public, m.sign, g)
	must(m.t, err)
	return token
}

// savedVideo is the one video `cozy run --out` saved.
func savedVideo(t *testing.T, out string, ran *bytes.Buffer) []byte {
	t.Helper()
	files, _ := filepath.Glob(filepath.Join(out, "*video*"))
	if len(files) != 1 {
		t.Fatalf("want one saved video in %s, got %v\n%s", out, files, ran.String())
	}
	saved, err := os.ReadFile(files[0])
	must(t, err)
	return saved
}

// A browser reaches a run's output on a cozy.machine.v1 machine over cozy/1, as it does on a Go
// machine: the receipt names the granted WebRTC port, a capability an authorized key signs
// follows the film as each revision lands, and the bytes are the ones `cozy run --out` saved.
// A finished output is read by range, and a capability for another output is refused.
func TestCozy1FollowsAJobsFilmOnAV1Machine(t *testing.T) {
	if *machineHostBinary == "" || *cpuLongform == "" {
		t.Skip("requires -machine-host=<cozy-machine> and -cpu-longform=<cozy-machine>/tests/fixtures/cpu_longform")
	}
	root, port := cozy1Root(t)
	project := filepath.Join(t.TempDir(), "cpu_longform")
	must(t, os.CopyFS(project, os.DirFS(*cpuLongform)))
	if out, err := exec.Command("uv", "lock", "--directory", project).CombinedOutput(); err != nil {
		t.Fatalf("uv lock: %v\n%s", err, out)
	}
	if code, out := runCozy(t, root, "package", "install", project, "--editable"); code != 0 {
		t.Fatalf("package install [exit %d]\n%s", code, out)
	}
	var encoded bytes.Buffer
	must(t, png.Encode(&encoded, image.NewRGBA(image.Rect(0, 0, 2, 2))))
	reference := filepath.Join(root, "ref.png")
	must(t, os.WriteFile(reference, encoded.Bytes(), 0o600))
	in, out := filepath.Join(root, "req.json"), filepath.Join(root, "film")
	must(t, os.WriteFile(in, []byte(`{"segments":["a dawn","a storm","a calm"],"hold":2}`), 0o600))
	run, command, ran := cozy1Run(t, root, "local/cozy-machine-cpu-longform/long_form",
		"--input", in, "--asset", "reference="+reference, "--await", "--json", "--out", out)
	h, open := cozy1Sessions(t, root, port, run)
	c := open(capability.Grant{})
	h.send(c, map[string]any{"t": "follow", "id": "f", "run": run, "output": "video", "after": 0, "offset": 0})
	var f mediaFollower
	h.until(c, &f, func() bool { return f.end != nil })
	if err := command.Wait(); err != nil {
		t.Fatalf("the job exited %v:\n%s", err, ran.String())
	}
	code, played := runCozy(t, root, "run", "play", run, "--output", "video", "--json")
	if code != 0 {
		t.Fatalf("ordinary fixture play [exit %d]: %s", code, played)
	}
	var link struct {
		Link string `json:"link"`
	}
	must(t, json.Unmarshal([]byte(lastJSONLine(played)), &link))
	_, fragment, ok := strings.Cut(link.Link, "#")
	if !ok {
		t.Fatal("play produced no fragment")
	}
	values, err := url.ParseQuery(fragment)
	must(t, err)
	if values.Get("r") != run || values.Get("o") != "video" || values.Get("a") == "" || len(values.Get("f")) != 64 {
		t.Fatalf("play lost its direct descriptor or request id: %s", link.Link)
	}
	host := machines.NewHost(filepath.Join(root, "machine"), "", nil)
	state, problem := host.Status()
	fatal(t, problem)
	owner, problem := host.ExistingOwner()
	fatal(t, problem)
	public, err := base64.RawURLEncoding.DecodeString(owner.PublicKey())
	must(t, err)
	granted, err := capability.Verify(values.Get("c"), state.MachineID, []ed25519.PublicKey{public}, time.Now(), "")
	must(t, err)
	if granted.Run != run || !slices.Equal(granted.Outputs, []string{"video"}) {
		t.Fatalf("play scope was changed: %+v", granted)
	}
	saved := savedVideo(t, out, ran)
	if !bytes.Equal(f.got, saved) || f.end.Status != "completed" || f.end.SHA256 != digestOf(saved) || f.end.Length != uint64(len(saved)) {
		t.Fatalf("the follower ended with %s holding %d bytes; --out saved %d", f.end.Raw, len(f.got), len(saved))
	}
	// One revision per segment, each a whole new film: a revision that lands after the
	// follower holds bytes resets it, and the last entry is the third revision.
	if len(f.entries) == 0 || f.resets > 2 || f.entries[len(f.entries)-1].Rev != 3 || f.entries[len(f.entries)-1].AppendedFrom != nil {
		t.Fatalf("entries %+v after %d resets", f.entries, f.resets)
	}

	h.send(c, map[string]any{"t": "get", "id": "g", "run": run, "output": "video", "offset": 3, "length": 10, "etag": "r3"})
	if got, end := h.body(c); !bytes.Equal(got, saved[3:13]) || end.SHA256 != digestOf(saved) {
		t.Fatalf("get answered %q then %s", got, end.Raw)
	}
	scoped := open(capability.Grant{Outputs: []string{"parts"}})
	h.send(scoped, map[string]any{"t": "follow", "id": "s", "run": run, "output": "video"})
	if refused := h.expect(scoped, "error"); refused.Code != "scope" {
		t.Fatalf("a capability for another output answered %s", refused.Raw)
	}
}

// A growing video reaches the follower as each segment lands: an appended revision sends only
// the bytes past the ones held, a replacing one resets, and a follower that left resumes at
// its cursor and ends with the same film `cozy run --out` saved.
func TestCozy1FollowsAGrowingVideoOnAV1Machine(t *testing.T) {
	if *machineHostBinary == "" || *privateScriptRuntimeWheel == "" {
		t.Skip("requires -machine-host=<cozy-machine> and -script-runtime-wheel")
	}
	wheel, err := filepath.Abs(*privateScriptRuntimeWheel)
	must(t, err)
	root, port := cozy1Root(t)
	if code, out := runCozy(t, root, "package", "install", outputLogProof(t, wheel), "--editable"); code != 0 {
		t.Fatalf("package install [exit %d]\n%s", code, out)
	}
	gate, out := t.TempDir(), filepath.Join(root, "film")
	run, command, ran := cozy1Run(t, root, "local/output-log-proof/film", "gate="+gate, "--await", "--json", "--out", out)
	h, open := cozy1Sessions(t, root, port, run)
	c := open(capability.Grant{Outputs: []string{"video"}})
	follow := func(c *webrtctest.Client, f *mediaFollower) {
		h.send(c, map[string]any{"t": "follow", "id": "f", "run": run, "output": "video", "after": f.seq, "offset": len(f.got)})
	}
	var f mediaFollower
	follow(c, &f)
	holds := func(c *webrtctest.Client, k int) {
		t.Helper()
		h.until(c, &f, func() bool {
			return slices.ContainsFunc(f.entries, func(entry webrtctest.Message) bool {
				return entry.Rev == uint64(k) && f.seq >= entry.Seq && uint64(len(f.got)) == entry.Length
			})
		})
	}
	must(t, os.WriteFile(filepath.Join(gate, "go-1"), nil, 0o600))
	holds(c, 1)
	first := append([]byte(nil), f.got...)
	must(t, os.WriteFile(filepath.Join(gate, "go-2"), nil, 0o600))
	holds(c, 2)
	second := f.entries[len(f.entries)-1]
	if appended := second.AppendedFrom; appended != nil && (f.resets != 0 || *appended != uint64(len(first)) || !bytes.HasPrefix(f.got, first)) {
		t.Fatalf("revision 2 appends from %d, yet the follower was reset %d times or its first %d bytes changed", *appended, f.resets, len(first))
	} else if appended == nil && f.resets != 1 {
		t.Fatalf("revision 2 replaces revision 1, yet the follower was reset %d times", f.resets)
	}
	// The follower leaves and another session resumes at its cursor: nothing held is sent again.
	must(t, c.Close())
	held, resets := len(f.got), f.resets
	resumed := open(capability.Grant{Outputs: []string{"video"}})
	follow(resumed, &f)
	must(t, os.WriteFile(filepath.Join(gate, "go-3"), nil, 0o600))
	h.until(resumed, &f, func() bool { return f.end != nil })
	if err := command.Wait(); err != nil {
		t.Fatalf("the job exited %v:\n%s", err, ran.String())
	}
	saved := savedVideo(t, out, ran)
	if !bytes.Equal(f.got, saved) || f.end.Status != "completed" || f.end.SHA256 != digestOf(saved) {
		t.Fatalf("the follower ended with %s holding %d bytes (%d before it resumed, %d resets since); --out saved %d",
			f.end.Raw, len(f.got), held, f.resets-resets, len(saved))
	}
}
