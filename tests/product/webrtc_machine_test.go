package producttest

import (
	"context"
	"crypto/ed25519"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"net"
	"net/http"
	"net/netip"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/capability"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/host/webrtc"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/secret"
	"github.com/cozy-creator/cozy/tests/product/webrtctest"
)

// grantWebRTC has the Hub grant machines it registers a WebRTC port, as it does for an image
// that serves browsers; the port is free when chosen.
func grantWebRTC(t *testing.T, h *machineHub) int {
	probe, err := net.Listen("tcp", "127.0.0.1:0") //cozy:allow choosing a free port for the grant
	must(t, err)
	port := probe.Addr().(*net.TCPAddr).Port
	must(t, probe.Close())
	h.grants = map[string]string{"COZY_WEBRTC_INTERNAL_PORT": strconv.Itoa(port)}
	return port
}

// machineListens is what a running machine Host shows of its WebRTC grant: the port its
// sealed receipt names (0: none), and every TCP port its process listens on beside its
// worker and media ports.
func machineListens(t *testing.T, dir string) (receipt int, others []int) {
	t.Helper()
	var record struct {
		PID        int `json:"pid"`
		WorkerPort int `json:"worker_port"`
		MediaPort  int `json:"media_port"`
	}
	raw, err := os.ReadFile(filepath.Join(dir, "host.json"))
	must(t, err)
	must(t, json.Unmarshal(raw, &record))
	var envelope struct {
		Payload []byte `json:"payload"`
	}
	var payload struct {
		WebRTC *struct {
			Port int `json:"port"`
		} `json:"webrtc"`
	}
	raw, err = os.ReadFile(filepath.Join(dir, "root/run/cozy/bootstrap/readiness-envelope.json"))
	must(t, err)
	must(t, json.Unmarshal(raw, &envelope))
	must(t, json.Unmarshal(envelope.Payload, &payload))
	if payload.WebRTC != nil {
		receipt = payload.WebRTC.Port
	}
	sockets := map[string]bool{}
	fds, err := os.ReadDir(fmt.Sprintf("/proc/%d/fd", record.PID))
	must(t, err)
	for _, fd := range fds {
		if link, err := os.Readlink(fmt.Sprintf("/proc/%d/fd/%s", record.PID, fd.Name())); err == nil && strings.HasPrefix(link, "socket:[") {
			sockets[strings.TrimSuffix(strings.TrimPrefix(link, "socket:["), "]")] = true
		}
	}
	for _, table := range []string{"tcp", "tcp6"} {
		raw, _ := os.ReadFile(fmt.Sprintf("/proc/%d/net/%s", record.PID, table))
		for _, line := range strings.Split(string(raw), "\n") {
			fields := strings.Fields(line)
			if len(fields) < 10 || fields[3] != "0A" || !sockets[fields[9]] { // 0A: LISTEN
				continue
			}
			_, hex, _ := strings.Cut(fields[1], ":")
			port, _ := strconv.ParseUint(hex, 16, 16)
			if p := int(port); p != record.WorkerPort && p != record.MediaPort {
				others = append(others, p)
			}
		}
	}
	slices.Sort(others)
	return receipt, others
}

// machineFollow follows one output of a machine over WebRTC, as a browser holding a link
// does: the leaf fingerprint, the port its receipt names, and a capability its owner key signs.
type machineFollow struct {
	t      *testing.T
	h      *mediaHarness
	c      *webrtctest.Client
	f      mediaFollower
	mint   func(capability.Grant) string
	origin string // the machine endpoint, for the same capability over HTTPS
	client *http.Client
}

func followOnMachine(t *testing.T, dir string, run uint64, output string, port int) *machineFollow {
	t.Helper()
	host := machines.NewHost(dir, "", nil)
	state, problem := host.Status()
	fatal(t, problem)
	owner, problem := host.Owner()
	fatal(t, problem)
	pin, problem := host.Pin()
	fatal(t, problem)
	raw, err := os.ReadFile(filepath.Join(dir, "leaf.pem"))
	must(t, err)
	leaf, _ := pem.Decode(raw)
	var record struct {
		WorkerPort int `json:"worker_port"`
	}
	raw, err = os.ReadFile(filepath.Join(dir, "host.json"))
	must(t, err)
	must(t, json.Unmarshal(raw, &record))
	public, err := base64.RawURLEncoding.DecodeString(owner.PublicKey())
	must(t, err)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	t.Cleanup(cancel)
	m := &machineFollow{t: t, h: &mediaHarness{t: t, ctx: ctx},
		origin: fmt.Sprintf("https://127.0.0.1:%d", record.WorkerPort),
		client: &http.Client{Transport: &http.Transport{TLSClientConfig: pin.TLSConfig()}},
		mint: func(g capability.Grant) string {
			g.Machine, g.Run, g.Expires = state.MachineID, strconv.FormatUint(run, 10), time.Now().Add(10*time.Minute).Unix()
			token, err := capability.MintSigned(ed25519.PublicKey(public), owner.Sign, g)
			must(t, err)
			return token
		}}
	m.c, err = webrtctest.Dial(ctx, netip.AddrPortFrom(netip.MustParseAddr("127.0.0.1"), uint16(port)), webrtc.Fingerprint(leaf.Bytes), webrtctest.Options{})
	must(t, err)
	t.Cleanup(func() { m.c.Close() })
	m.h.send(m.c, map[string]any{"t": "hello", "v": 1, "cap": m.mint(capability.Grant{})})
	m.h.expect(m.c, "welcome")
	m.h.send(m.c, map[string]any{"t": "credit", "bytes": 1 << 30})
	m.h.send(m.c, map[string]any{"t": "follow", "id": "f", "run": strconv.FormatUint(run, 10), "output": output, "after": 0, "offset": 0})
	return m
}

// revision waits until the follower holds revision k whole.
func (m *machineFollow) revision(k int) {
	m.t.Helper()
	m.h.until(m.c, &m.f, func() bool { return len(m.f.entries) >= k && uint64(len(m.f.got)) == m.f.entries[k-1].Length })
}

// https reads the output with a capability over the machine endpoint and answers the status.
func (m *machineFollow) https(run uint64, output, token string) int {
	request, err := http.NewRequest(http.MethodGet, fmt.Sprintf("%s/v1/runs/%d/outputs/%s", m.origin, run, output), nil)
	must(m.t, err)
	request.Header.Set("Authorization", "Cozy-Cap "+token)
	response, err := m.client.Do(request)
	must(m.t, err)
	response.Body.Close()
	return response.StatusCode
}

// A rental granted the WebRTC port serves its film to a browser holding a link, over the real
// daemon, Host and Runtime. Its receipt names the port, the only one it listens on beside its
// endpoint's. Each segment reaches the follower as it lands, and the end carries the film's
// digest. HTTPS refuses the capability bound to the follower's DTLS certificate.
func TestWebRTCServesARentalsFilmAsItLands(t *testing.T) {
	integration(t)
	if *privateScriptRuntimeWheel == "" || *machineHostBinary == "" {
		t.Skip("requires -script-runtime-wheel and -machine-host: a rental's Host and a real Runtime")
	}
	uv, err := exec.LookPath("uv")
	if err != nil {
		t.Skip("uv lays out the machine root")
	}
	wheel, err := filepath.Abs(*privateScriptRuntimeWheel)
	must(t, err)
	h := newMachineHub(t)
	port := grantWebRTC(t, h)
	root, err := os.MkdirTemp("", "czm")
	must(t, err)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+h.server.URL+
		"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "down")
		if !t.Failed() {
			_ = removeAllForce(root)
		}
	})
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	source := machines.Source{Host: *machineHostBinary, RuntimeWheel: *machineRuntimeWheel, TensorFSWheel: *machineTensorFSWheel}
	launch, identity, token, provider := providerHost(t, h, layout, source, uv)
	dir := filepath.Dir(provider)
	if receipt, others := machineListens(t, dir); receipt != port || !slices.Equal(others, []int{port}) {
		t.Fatalf("a machine granted WebRTC port %d names %d in its receipt and also listens on %v", port, receipt, others)
	}
	h.mu.Lock()
	h.rentals[parityRental] = map[string]any{"rental_id": parityRental, "name": "tessa", "state": "ready",
		"requested_accelerator_model": "Virtual Accelerator", "accelerator_count": 4, "hourly_rate_usd_micros": 1,
		"worker_address": launch.Addr, "media_address": launch.MediaAddr}
	h.mu.Unlock()
	fatal(t, rental.Attach(layout, store, records.Rental{ID: parityRental, MachineName: "tessa", SKU: "virtual-4", State: "ready",
		AcceleratorModel: "Virtual Accelerator", AcceleratorCount: 4, HourlyRateUSDMicros: 1, Hub: h.server.URL,
		Address: launch.Addr, MediaAddress: launch.MediaAddr, ExpectedWorkerID: launch.WorkerID, ExpectedWorkerBootID: launch.BootID},
		string(pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: launch.Leaf})), secret.New(token), identity))
	if code, out := runCozy(t, root, "package", "install", outputLogProof(t, wheel), "--editable"); code != 0 {
		t.Fatalf("editable install [exit %d]\n%s", code, out)
	}
	gate := t.TempDir()
	command := exec.Command("/usr/bin/nice", "-n", "19", cozyBin, "run", "local/output-log-proof/film", "gate="+gate, "--rental=tessa", "--await")
	command.Env = childEnv(t, root)
	var filmed strings.Builder
	command.Stdout, command.Stderr = &filmed, &filmed
	must(t, command.Start())
	products := func() int {
		requests, _ := store.Requests("", "local/output-log-proof", 10)
		for _, row := range requests {
			if list, _ := store.Products(row.ID); len(list) > 0 {
				return len(list)
			}
		}
		return 0
	}
	// The rental's first run is its number 1. Its first segment proves it is on the machine.
	must(t, os.WriteFile(filepath.Join(gate, "go-1"), nil, 0o600))
	landed(t, "the film's first segment", func() bool { return products() == 1 })
	media := followOnMachine(t, dir, 1, "video", port)
	media.revision(1)
	for k := 2; k <= 3; k++ {
		must(t, os.WriteFile(filepath.Join(gate, fmt.Sprintf("go-%d", k)), nil, 0o600))
		media.revision(k) // each segment reaches the browser as it lands
		if got := countFrames(t, media.f.got); got != strconv.Itoa(12*k) {
			t.Fatalf("after segment %d the WebRTC follower's copy decodes %s frames", k, got)
		}
	}
	if err := command.Wait(); err != nil {
		t.Fatalf("the film run exited %v:\n%s", err, filmed.String())
	}
	media.h.until(media.c, &media.f, func() bool { return media.f.end != nil })
	if end := media.f.end; end.Status != "completed" || end.SHA256 != digestOf(media.f.got) || media.f.resets != 0 {
		t.Fatalf("the WebRTC follower ended with %s after %d bytes and %d resets", end.Raw, len(media.f.got), media.f.resets)
	}
	if code := media.https(1, "video", media.mint(capability.Grant{Binding: media.c.Cert})); code != http.StatusForbidden {
		t.Fatalf("a bound capability over HTTPS answered %d", code)
	}
	if code := media.https(1, "video", media.mint(capability.Grant{})); code != http.StatusOK {
		t.Fatalf("the unbound capability over HTTPS answered %d", code)
	}
}

// A machine granted no WebRTC port listens on nothing but its worker and media ports, and
// its receipt names no webrtc member.
func TestWebRTCListensOnlyWhenGranted(t *testing.T) {
	if *machineHostBinary == "" {
		t.Skip("requires -machine-host: the Host a rental runs")
	}
	uv, err := exec.LookPath("uv")
	if err != nil {
		t.Skip("uv lays out the machine root")
	}
	h := newMachineHub(t)
	root, err := os.MkdirTemp("", "czw")
	must(t, err)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+h.server.URL+
		"\ntensorhub_token: rental-idle-test\n"), 0o600))
	t.Cleanup(func() {
		if !t.Failed() {
			_ = removeAllForce(root)
		}
	})
	layout, problem := home.Open(root)
	fatal(t, problem)
	source := machines.Source{Host: *machineHostBinary, RuntimeWheel: *machineRuntimeWheel, TensorFSWheel: *machineTensorFSWheel}
	_, _, _, provider := providerHost(t, h, layout, source, uv)
	if receipt, others := machineListens(t, filepath.Dir(provider)); receipt != 0 || len(others) != 0 {
		t.Fatalf("a machine granted no WebRTC port names %d in its receipt and listens on %v", receipt, others)
	}
}
