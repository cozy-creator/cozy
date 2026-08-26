package main

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/json"
	"encoding/pem"
	"math/big"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy-creator-v2/internal/secret"
)

// podHub is the SECOND independent implementation of the hub's rental contract (cl-015),
// standing where the real hub stands: it provisions, it hands back a dial triple, it tears
// down. It exists for the same reason `fakeworker` does — the client's rental path must be
// exercised against a peer it did not co-develop with, over the committed routes alone —
// and for one more: a pod costs money, and every arm below is a leg this driver can run a
// hundred times on a laptop before a card is ever billed.
//
// What it provisions is a REAL worker: `cozy-live fakeworker --arm remote` behind TLS with
// a certificate it mints, holding the owner token in `COZY_BOOTSTRAP_CREDENTIAL` — the
// exact contract a rented pod's `cozy-runtime serve --tls-cert --tls-key` keeps. So the
// leg this proves is the whole owner side: dial the address, trust exactly that PEM,
// present the owner token as Claim.proof, converse, dispatch, settle.
//
// What it is NOT: a provider. It boots nothing, bills nothing, and knows no card names —
// `card` is carried and echoed, never interpreted.

type podRental struct {
	ID      string
	State   string
	Address string
	CertPEM string
	Token   string
	PodID   string
	Detail  string
	// Media is where this pod's co-resident MEDIA SERVER answers (cl-014). A real hub
	// names it in the rental document; this stand-in names it too, so the client's
	// verbatim read of the field is exercised as well as its derivation fallback.
	Media string
	// MediaRoot is the subtree that media server owns on the pod's filesystem — the
	// arms read it directly to see where bytes actually landed.
	MediaRoot string
	// The two CO-RESIDENT processes, separately. cl-014's coupling claim is that neither
	// is in the other's request path, and the way to observe that is to kill one.
	WorkerPID int
	MediaPID  int
}

type podHub struct {
	mu    sync.Mutex
	token string // the admin credential this hub admits, verbatim
	dir   string // scratch: certs, keys, the spawned worker's run root
	// pods is the POD SIDE's filesystem root, and it is deliberately NOT the client's.
	// Handing the "pod" the LocalService's own root made every client-local `file://`
	// grant resolve by accident: the byte boundary this section exists to exercise was
	// not there at all, and the arms over it were green about nothing.
	pods string
	// arm is what the provisioned worker plays. The pod side is a peer with its own
	// behaviour, so the hub picks it once and every pod it provisions runs it.
	arm     string
	rentals map[string]*podRental
	deleted map[string]bool
	// fail makes every pod this hub provisions end `failed`. It is the arm for a rental
	// that never comes up, which a client must answer for rather than wait out.
	fail bool
	// noMedia provisions a pod with a worker and NO media server: a control leg that can
	// be dialled and a pod that cannot be handed a byte. It is the arm for the refusal
	// this client owes instead of falling back to granting paths on its own disk.
	noMedia bool
	// release is what the provisioned worker DECLARES on ClaimAck. A real hub installs the
	// release the renter asked for and the pod says which; this stand-in is told, so the
	// owner's release pin has both a matching pod to accept and a lying one to refuse
	// (#505's carried-not-verified gap).
	releaseID string
	procs     []*exec.Cmd
	addr      string
	ln        net.Listener
}

// podHubSpec is what a stand-in hub is told to provision. Everything in it is a fact a
// REAL hub knows and this one has to be given: which release it installed on the pod, what
// the pod's worker plays, and whether this pod is one of the broken shapes an owner must
// answer for.
type podHubSpec struct {
	Dir     string
	Arm     string
	Release string
	Fail    bool // every pod ends `failed`: the rental that never comes up
	NoMedia bool // a worker with no co-resident media server: a pod that cannot be fed
}

func startPodHub(dir, arm string, fail bool) *podHub {
	return newPodHub(podHubSpec{Dir: dir, Arm: arm, Fail: fail})
}

func newPodHub(spec podHubSpec) *podHub {
	dir, arm := spec.Dir, spec.Arm
	must("creating the hub scratch", os.MkdirAll(dir, 0o755))
	h := &podHub{
		token: randomHex(16), // an opaque admin credential; its shape is the hub's
		dir:   dir, pods: filepath.Join(dir, "pod-fs"), arm: arm,
		rentals: map[string]*podRental{}, deleted: map[string]bool{}, fail: spec.Fail,
		noMedia: spec.NoMedia, releaseID: spec.Release,
	}
	must("creating the pod-side filesystem", os.MkdirAll(h.pods, 0o755))
	ln, err := net.Listen("tcp", "127.0.0.1:0") //cozy:allow the DRIVER hosts a stand-in hub; the product binds through internal/api
	must("binding the fake hub", err)
	h.ln, h.addr = ln, ln.Addr().String()
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/rentals", h.rent)
	mux.HandleFunc("GET /v1/rentals/{id}", h.read)
	mux.HandleFunc("DELETE /v1/rentals/{id}", h.release)
	go func() { _ = http.Serve(ln, mux) }()
	return h
}

// url is what TENSORHUB_URL is set to for the CLI under test.
func (h *podHub) url() string { return "http://" + h.addr }

// env is the pair a `cozy rent` invocation carries: this hub and its admin credential.
func (h *podHub) env() []string {
	return []string{"TENSORHUB_URL=" + h.url(), "TENSORHUB_TOKEN=" + h.token}
}

func (h *podHub) close() {
	h.mu.Lock()
	procs := h.procs
	h.procs = nil
	h.mu.Unlock()
	for _, p := range procs {
		if p.Process != nil {
			_ = killGroup(p.Process.Pid, syscall.SIGKILL)
		}
	}
	if h.ln != nil {
		_ = h.ln.Close()
	}
}

// released answers whether the hub ever saw a DELETE for this rental — the arm's evidence
// that `cozy rent release` reached the provider rather than only forgetting locally.
func (h *podHub) released(id string) bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.deleted[id]
}

func (h *podHub) live() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return len(h.rentals)
}

// admits is the credential check, spelled the way the real hub's is: a first-party route
// with the wrong bearer is 401 with a TYPED envelope, and the client's exit code is the
// matrix's, not this hub's status.
func (h *podHub) admits(w http.ResponseWriter, r *http.Request) bool {
	presented := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
	if strings.TrimSpace(presented) == h.token {
		return true
	}
	refuse(w, http.StatusUnauthorized, "rental.unauthenticated",
		"this hub does not admit the presented credential",
		"set TENSORHUB_TOKEN to this hub's admin token")
	return false
}

func refuse(w http.ResponseWriter, status int, code, message, remedy string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]string{
		"code": code, "message": message, "remedy": remedy,
	}})
}

func (h *podHub) rent(w http.ResponseWriter, r *http.Request) {
	if !h.admits(w, r) {
		return
	}
	if r.Header.Get("X-Tensorhub-Reason") == "" {
		refuse(w, http.StatusBadRequest, "reason_required",
			"a mutation is recorded with a reason before it happens", "pass --reason <why>")
		return
	}
	var body struct {
		Endpoint string `json:"endpoint"`
		Card     string `json:"card"`
		Hint     int    `json:"duration_hint_s"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body.Endpoint == "" || body.Card == "" {
		refuse(w, http.StatusBadRequest, "invalid_request",
			"a rental names an endpoint and a card", `{"endpoint":"h3","card":"H200"}`)
		return
	}
	id := "rnt-" + randomHex(8)
	rec := &podRental{ID: id, State: "provisioning", PodID: "pod-" + randomHex(6),
		Detail: "asking the provider for " + body.Card}
	h.mu.Lock()
	h.rentals[id] = rec
	h.mu.Unlock()
	go h.provision(rec)

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusAccepted)
	_ = json.NewEncoder(w).Encode(map[string]any{"rental_id": id, "state": rec.State})
}

// provision mints the pod's identity and starts the worker that hosts it. The real hub
// does this on a machine it rented; the shape of what it hands back is identical.
func (h *podHub) provision(rec *podRental) {
	if h.fail {
		h.mu.Lock()
		rec.State, rec.Detail = "failed", "no capacity for this card in any region"
		h.mu.Unlock()
		return
	}
	certPath := filepath.Join(h.dir, rec.ID+".pem")
	keyPath := filepath.Join(h.dir, rec.ID+".key")
	certPEM, err := writeSelfSigned(certPath, keyPath)
	if err != nil {
		h.mu.Lock()
		rec.State, rec.Detail = "failed", "cannot mint the worker's certificate: "+err.Error()
		h.mu.Unlock()
		return
	}
	// The owner token is the pod's OWN provisioned credential, minted here and imposed on
	// the worker through the one channel only it inherits — the same handoff a spawned
	// local worker's bootstrap credential rides (#449/#463).
	token := randomHex(32)
	runRoot := filepath.Join(h.dir, rec.ID+"-run")
	must("creating the pod run root", os.MkdirAll(runRoot, 0o755))

	self, err := os.Executable()
	if err != nil {
		h.mu.Lock()
		rec.State, rec.Detail = "failed", "no driver binary: "+err.Error()
		h.mu.Unlock()
		return
	}
	// THE POD'S OWN ROOT, per rental. Nothing the client wrote is reachable from it and
	// nothing written here is reachable from the client — which is the only way an arm
	// over the byte boundary can observe anything.
	podRoot := filepath.Join(h.pods, rec.ID)
	must("creating the pod filesystem", os.MkdirAll(podRoot, 0o755))
	args := []string{"fakeworker", "--arm", h.arm,
		"--socket", "127.0.0.1:0", "--out", runRoot,
		"--tls-cert", certPath, "--tls-key", keyPath,
		"--cozy-home", podRoot}
	if h.releaseID != "" {
		// WHAT THIS POD SERVES, declared on ClaimAck. A hub that installed the release
		// knows it; a pod that will not say is refused by the owner's pin.
		args = append(args, "--release-id", h.releaseID)
	}
	cmd := niceCmd(self, args...)
	cmd.Env = append(childEnv(podRoot), "COZY_BOOTSTRAP_CREDENTIAL="+token)
	logPath := filepath.Join(runRoot, "pod-worker.log")
	logFile, err := os.Create(logPath)
	must("the pod worker log", err)
	cmd.Stdout, cmd.Stderr = logFile, logFile
	setProcessGroup(cmd)
	if err := cmd.Start(); err != nil {
		h.mu.Lock()
		rec.State, rec.Detail = "failed", "the pod's worker did not start: "+err.Error()
		h.mu.Unlock()
		return
	}
	h.mu.Lock()
	h.procs = append(h.procs, cmd)
	rec.WorkerPID = cmd.Process.Pid
	rec.Detail = "the pod is up; waiting for its worker to bind"
	h.mu.Unlock()

	// THE SECOND PROCESS IN THE POD'S CONTAINER (cl-014): the media server, co-resident
	// with the worker, holding its OWN keys and coupled to the worker BY THE FILESYSTEM.
	// It is what the owner uploads a payload and a binding record to and downloads an
	// output from, and it is started here because provisioning a pod is the hub's job.
	//
	// Its credential is the rental's own owner token, delivered as a HASH: the provisioner
	// knows the token because it minted it, and the server that checks it never does.
	mediaAddr, mediaRoot := "", ""
	if !h.noMedia {
		mediaRoot = filepath.Join(podRoot, "media")
		tokenFile := filepath.Join(runRoot, "media.tokens")
		if err := os.WriteFile(tokenFile,
			[]byte(secret.HashLine(secret.New(token))+"\n"), 0o600); err != nil {
			h.mu.Lock()
			rec.State, rec.Detail = "failed", "cannot provision the media credential: "+err.Error()
			h.mu.Unlock()
			return
		}
		media := niceCmd(mediaBinary(),
			"--listen", "127.0.0.1:0",
			"--root", mediaRoot,
			"--plans", filepath.Join(podRoot, "binding-plans"),
			"--tokens", tokenFile,
			"--out", runRoot,
			"--tls-cert", certPath, "--tls-key", keyPath,
			// A small quota on purpose: the arm that fills it is cheap, and a media
			// subtree is separately bounded so an upload can never ENOSPC the journal.
			"--quota", "4194304")
		mediaLog, err := os.Create(filepath.Join(runRoot, "pod-media.log"))
		must("the pod media log", err)
		media.Stdout, media.Stderr = mediaLog, mediaLog
		setProcessGroup(media)
		if err := media.Start(); err != nil {
			h.mu.Lock()
			rec.State, rec.Detail = "failed", "the pod's media server did not start: "+err.Error()
			h.mu.Unlock()
			return
		}
		h.mu.Lock()
		h.procs = append(h.procs, media)
		rec.MediaPID = media.Process.Pid
		h.mu.Unlock()
		mediaAddr = awaitAddr(filepath.Join(runRoot, "media.addr"), media)
		if mediaAddr == "" {
			h.mu.Lock()
			rec.State, rec.Detail = "failed",
				"the pod's media server exited before binding; log "+filepath.Join(runRoot, "pod-media.log")
			h.mu.Unlock()
			return
		}
	}

	// READY IS AN OBSERVATION, never a timer: the hub says ready when the worker has
	// PUBLISHED an address, which is the same file-handoff discovery contract a locally
	// spawned worker keeps (#436). A worker that dies instead is the answer.
	addrFile := filepath.Join(runRoot, "control.addr")
	for {
		if data, err := os.ReadFile(addrFile); err == nil && len(data) > 0 {
			h.mu.Lock()
			rec.Address = strings.TrimSpace(string(data))
			rec.CertPEM, rec.Token, rec.State = certPEM, token, "ready"
			rec.Media, rec.MediaRoot = mediaAddr, mediaRoot
			rec.Detail = "the worker is hosting WorkerControl behind TLS"
			h.mu.Unlock()
			return
		}
		if cmd.ProcessState != nil {
			h.mu.Lock()
			rec.State, rec.Detail = "failed", "the pod's worker exited before binding; log "+logPath
			h.mu.Unlock()
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// awaitAddr reads a bound address out of the file the process publishes. The wait is
// bounded by the PROCESS: one that dies instead of binding is the answer, and there is no
// clock here about how long binding a socket ought to take.
func awaitAddr(path string, cmd *exec.Cmd) string {
	for {
		if data, err := os.ReadFile(path); err == nil && len(data) > 0 {
			return strings.TrimSpace(string(data))
		}
		if cmd.ProcessState != nil {
			return ""
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// mediaBinary is the pod-side media server this driver provisions. It is the real built
// binary, not a stand-in: the point of the section is that a REAL byte plane carried the
// bytes, and a fake one would prove the owner talks to itself.
func mediaBinary() string {
	abs, err := filepath.Abs(flag("media", "./cozy-media"))
	must("resolving the cozy-media binary", err)
	return abs
}

// killPodWorker ends ONE pod's worker process and leaves its media server running. It is
// how cl-014's coupling claim becomes an observation: the two processes share files and
// nothing else, so a dead worker must not take the byte plane down with it.
func (h *podHub) killPodWorker(id string) int {
	h.mu.Lock()
	rec := h.rentals[id]
	pid := 0
	if rec != nil {
		pid = rec.WorkerPID
	}
	var victim *exec.Cmd
	for _, p := range h.procs {
		if p.Process != nil && p.Process.Pid == pid {
			victim = p
		}
	}
	h.mu.Unlock()
	if pid <= 0 {
		return 0
	}
	_ = killGroup(pid, syscall.SIGKILL)
	// AND IT IS REAPED. A killed child this process started stays a zombie until someone
	// waits on it, and a zombie answers signal 0 — so an arm asking "is it gone" would be
	// told yes-it-is-still-there about a process that is already dead. The hub started it,
	// so the hub collects it.
	if victim != nil {
		_ = victim.Wait()
	}
	return pid
}

// mediaOf is a rental's media address, so an arm can dial the pod's byte plane directly.
func (h *podHub) mediaOf(id string) string {
	h.mu.Lock()
	defer h.mu.Unlock()
	if rec := h.rentals[id]; rec != nil {
		return rec.Media
	}
	return ""
}

// mediaRootOf is where that server keeps what it was handed — the pod-side directory an
// arm walks to see that bytes really crossed.
func (h *podHub) mediaRootOf(id string) string {
	h.mu.Lock()
	defer h.mu.Unlock()
	if rec := h.rentals[id]; rec != nil {
		return rec.MediaRoot
	}
	return ""
}

func (h *podHub) read(w http.ResponseWriter, r *http.Request) {
	if !h.admits(w, r) {
		return
	}
	h.mu.Lock()
	rec := h.rentals[r.PathValue("id")]
	if rec == nil {
		h.mu.Unlock()
		refuse(w, http.StatusNotFound, "rental.not_found",
			"no rental "+r.PathValue("id")+" on this hub", "`cozy rent` mints one")
		return
	}
	out := map[string]any{
		"rental_id": rec.ID, "state": rec.State, "address": rec.Address,
		"cert_pem": rec.CertPEM, "owner_token": rec.Token,
		"pod_id": rec.PodID, "detail": rec.Detail,
		// The one field this lane adds to the consumed contract (#506b). A hub that omits
		// it gets the client's stated convention instead; naming it here exercises the
		// verbatim path, which is the one a real hub will take once th-041 carries it.
		"media_address": rec.Media,
	}
	h.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(out)
}

func (h *podHub) release(w http.ResponseWriter, r *http.Request) {
	if !h.admits(w, r) {
		return
	}
	id := r.PathValue("id")
	h.mu.Lock()
	rec := h.rentals[id]
	if rec == nil {
		h.mu.Unlock()
		refuse(w, http.StatusNotFound, "rental.not_found",
			"no rental "+id+" to release on this hub", "`cozy rent ls` names what this host holds")
		return
	}
	delete(h.rentals, id)
	h.deleted[id] = true
	procs := h.procs
	h.mu.Unlock()
	// The pod is DESTROYED, which for this stand-in means the worker process it hosted
	// stops answering. An arm that dialled it afterwards must find nothing.
	for _, p := range procs {
		if p.Process != nil {
			_ = killGroup(p.Process.Pid, syscall.SIGKILL)
		}
	}
	w.WriteHeader(http.StatusNoContent)
}

// writeSelfSigned mints the certificate the rental hands over to be PINNED. It is signed
// by nothing: trusting exactly this PEM and no CA is what makes the pin a pin, so a
// self-signed leaf is the honest shape rather than a shortcut.
func writeSelfSigned(certPath, keyPath string) (string, error) {
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return "", err
	}
	serial, err := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 128))
	if err != nil {
		return "", err
	}
	template := x509.Certificate{
		SerialNumber:          serial,
		Subject:               pkix.Name{CommonName: "cozy-pod-worker"},
		NotBefore:             time.Now().Add(-time.Minute),
		NotAfter:              time.Now().Add(24 * time.Hour),
		KeyUsage:              x509.KeyUsageDigitalSignature | x509.KeyUsageCertSign,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
		IsCA:                  true,
		// The owner dials an ADDRESS, so the name the certificate must carry is that
		// address's host. A pod with a hostname carries it here instead.
		IPAddresses: []net.IP{net.ParseIP("127.0.0.1"), net.ParseIP("::1")},
		DNSNames:    []string{"localhost"},
	}
	der, err := x509.CreateCertificate(rand.Reader, &template, &template, &key.PublicKey, key)
	if err != nil {
		return "", err
	}
	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der})
	if err := os.WriteFile(certPath, certPEM, 0o644); err != nil {
		return "", err
	}
	keyDER, err := x509.MarshalECPrivateKey(key)
	if err != nil {
		return "", err
	}
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyDER})
	if err := os.WriteFile(keyPath, keyPEM, 0o600); err != nil {
		return "", err
	}
	return string(certPEM), nil
}

// tokenOf reads back the token this hub issued for one rental, so an arm can assert that
// the value NEVER appeared in the client's output. Checking that a credential was not
// printed requires knowing the credential.
func (h *podHub) tokenOf(id string) string {
	h.mu.Lock()
	defer h.mu.Unlock()
	if rec := h.rentals[id]; rec != nil {
		return rec.Token
	}
	return ""
}

// holds answers whether this hub still has the rental — which is how the teardown pass
// decides who to ask, when several stand-in hubs are live at once.
func (h *podHub) holds(id string) bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.rentals[id] != nil
}
