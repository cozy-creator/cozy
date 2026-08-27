// cozy-media is THE POD'S MEDIA SERVER (cl-014, ruled #506b): the byte plane of a rented
// pod, and the only way bytes cross between an owner and the worker it rented.
//
// It is a SECOND PROCESS in the pod's single container, co-resident with the Python worker
// and coupled to it BY THE FILESYSTEM ALONE. There is no RPC between the two in either
// direction — the worker never calls this server and this server never calls the worker —
// so this process dying loses no attempt state and a killed worker does not stop an owner
// from downloading what it already produced. That is a structural property here, not a
// discipline: this binary imports no protocol client and, by fence, contains no outbound
// network call of any kind. It joins the liability fence by HAVING NOTHING TO EGRESS WITH.
//
// Six routes, and the shape of each is the whole design:
//
//	GET  /v1/bootstrap/receipt    Tensorhub reads one attempt-bound, pod-authored readiness
//	                              receipt. It has its OWN bearer hash, which cannot authorize
//	                              any renter route. The response is the exact file bytes.
//	PUT  /v1/inputs/{blob}        the owner uploads one attempt input; the answer is the
//	                              POD-LOCAL PATH it landed at, which is what the owner then
//	                              mints into the DeliveryGrant. The owner never guesses a
//	                              pod path and this server never learns an owner path.
//	PUT  /v1/plans/{plan-id}      the owner delivers one binding-plan record into the
//	                              worker's own `binding-plans` directory. FAIL-CLOSED: the
//	                              record's identity is re-hashed HERE and must equal the id
//	                              it was delivered under. That check exists only because
//	                              #506a made the identity path-free — under the old shape
//	                              the pod could not have recomputed it at all.
//	POST /v1/outputs/{slot}       the owner reserves one attempt's output directory AND its
//	                              exact maximum byte budget, then is told the pod-local path
//	                              to grant the worker.
//	GET  /v1/outputs/{slot}/{id}  the owner reads back what the worker wrote. READ-ONLY:
//	                              this route opens a file and never creates one.
//	DELETE /v1/attempts/{slot}    after rollback, or after mirror plus outcome ack, the
//	                              owner drops exactly that attempt's inputs and outputs.
//
// AUTH is a bearer against a TOKEN-HASH FILE, stat-per-request and fail-closed: the file is
// re-read whenever its (size, mtime) changes, an unreadable or empty file authenticates
// NOBODY, and this process never holds the token — only its digest. The epoch never moves
// backward: a replacement whose mtime predates what is loaded is refused rather than
// applied, so a restored backup cannot reinstate a retired credential.
//
// The SUBTREE IS QUOTA-BOUNDED and it is not the worker's root: uploads can never ENOSPC
// the journal. The worker writes outputs through the shared filesystem, so this process
// durably reserves their full grant bounds before returning a directory. HTTP uploads and
// direct worker writes thereby consume one media-owned budget without a cross-process lock.
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/cozy-creator/cozy-creator-v2/internal/plan"
	"github.com/cozy-creator/cozy-creator-v2/internal/secret"
)

func main() {
	os.Exit(run(os.Args[1:]))
}

// options is the whole launch surface. Everything is a path grant or a bound; nothing is
// discovered, nothing is read from the environment, and there is no configuration file.
type options struct {
	listen             string
	root               string // the quota-bounded subtree this server OWNS
	plans              string // the worker's `binding-plans` directory — write-only, from here
	tokens             string // the token-hash file, `sha256:<64 hex>` one per line
	out                string // where `media.addr` is published (the file-handoff discovery contract)
	cert               string
	key                string
	bootstrapReceipt   string // exact pod-authored JSON; may appear after the listener starts
	bootstrapTokenHash string // separate one-attempt verifier; never authorizes renter routes
	quota              int64
	maxBody            int64
}

func run(args []string) int {
	opt := options{quota: 8 << 30, maxBody: 256 << 20}
	for i := 0; i < len(args); i++ {
		name := strings.TrimPrefix(args[i], "--")
		value := ""
		if i+1 < len(args) && !strings.HasPrefix(args[i+1], "--") {
			value, i = args[i+1], i+1
		}
		switch name {
		case "listen":
			opt.listen = value
		case "root":
			opt.root = value
		case "plans":
			opt.plans = value
		case "tokens":
			opt.tokens = value
		case "out":
			opt.out = value
		case "tls-cert":
			opt.cert = value
		case "tls-key":
			opt.key = value
		case "bootstrap-receipt":
			opt.bootstrapReceipt = value
		case "bootstrap-token-sha256":
			opt.bootstrapTokenHash = value
		case "quota":
			n, err := strconv.ParseInt(value, 10, 64)
			if err != nil || n <= 0 {
				return usage("--quota %q is not a positive byte count", value)
			}
			opt.quota = n
		case "max-body":
			n, err := strconv.ParseInt(value, 10, 64)
			if err != nil || n <= 0 {
				return usage("--max-body %q is not a positive byte count", value)
			}
			opt.maxBody = n
		default:
			return usage("unknown flag %q", args[i])
		}
	}
	switch {
	case opt.listen == "":
		return usage("--listen <host:port> is required")
	case opt.root == "":
		return usage("--root <dir> is required: this server owns exactly one subtree")
	case opt.tokens == "":
		return usage("--tokens <file> is required: there is no unauthenticated mode")
	case (opt.bootstrapReceipt == "") != (opt.bootstrapTokenHash == ""):
		return usage("--bootstrap-receipt and --bootstrap-token-sha256 must be supplied together")
	case opt.bootstrapTokenHash != "" && !validTokenHash(opt.bootstrapTokenHash):
		return usage("--bootstrap-token-sha256 must be sha256:<64 lowercase hex>")
	}
	for _, dir := range []string{filepath.Join(opt.root, "inputs"), filepath.Join(opt.root, "outputs")} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fatal("cannot create %s: %v", dir, err)
		}
	}
	reservationDir := filepath.Join(opt.root, ".reservations")
	if err := os.MkdirAll(reservationDir, 0o700); err != nil {
		return fatal("cannot create the media reservation directory: %v", err)
	}
	// A crash before an atomic reservation rename can leave only its hidden temporary.
	// No grant was returned in that state, so boot removes it before quota is admitted.
	if entries, err := os.ReadDir(reservationDir); err != nil {
		return fatal("cannot inspect the media reservation directory: %v", err)
	} else {
		for _, entry := range entries {
			if strings.HasPrefix(entry.Name(), ".") && strings.HasSuffix(entry.Name(), ".staging") {
				if err := os.Remove(filepath.Join(reservationDir, entry.Name())); err != nil {
					return fatal("cannot remove interrupted media reservation %s: %v", entry.Name(), err)
				}
			}
		}
	}
	if opt.plans != "" {
		if err := os.MkdirAll(opt.plans, 0o755); err != nil {
			return fatal("cannot create the binding-plan directory %s: %v", opt.plans, err)
		}
	}
	s := &server{opt: opt}
	// FAIL CLOSED AT BOOT, not at the first request: a media server whose token file is
	// absent or empty admits nobody, and saying so now beats answering 401 to the owner who
	// just rented the pod and has no way to see why.
	if err := s.reloadTokens(); err != nil {
		return fatal("the token-hash file %s is unusable: %v — a media server with no "+
			"credential to check authenticates nobody", opt.tokens, err)
	}
	return s.serve()
}

type server struct {
	opt options

	mu     sync.Mutex
	writes sync.Mutex // quota admission and the filesystem mutation are one critical section
	// hashes is the loaded credential set: `sha256:<64 hex>` lines, never a raw token.
	hashes []string
	// size/stamp are what the loaded set was read from. Stat-per-request compares against
	// them, so a rotation is picked up on the next call without a restart and without a
	// re-read on every call.
	size  int64
	stamp time.Time
}

// reloadTokens re-reads the hash file if and only if its (size, mtime) moved, and refuses
// a move BACKWARD. Everything about it is fail-closed: an unreadable file, a file with no
// usable line, or a stamp older than the loaded one leaves the previous set in place and
// returns an error the caller turns into a refusal.
func (s *server) reloadTokens() error {
	info, err := os.Stat(s.opt.tokens)
	if err != nil {
		return err
	}
	s.mu.Lock()
	same := info.Size() == s.size && info.ModTime().Equal(s.stamp)
	older := !s.stamp.IsZero() && info.ModTime().Before(s.stamp)
	s.mu.Unlock()
	if same {
		return nil
	}
	if older {
		return fmt.Errorf("its mtime %s predates the loaded %s: the credential epoch is "+
			"monotone and never moves backward", info.ModTime().UTC().Format(time.RFC3339),
			s.stamp.UTC().Format(time.RFC3339))
	}
	data, err := os.ReadFile(s.opt.tokens)
	if err != nil {
		return err
	}
	var lines []string
	for _, line := range strings.Split(string(data), "\n") {
		if line = strings.TrimSpace(line); strings.HasPrefix(line, "sha256:") {
			lines = append(lines, line)
		}
	}
	if len(lines) == 0 {
		return errors.New("it holds no `sha256:<64 hex>` line")
	}
	s.mu.Lock()
	s.hashes, s.size, s.stamp = lines, info.Size(), info.ModTime()
	s.mu.Unlock()
	return nil
}

// admits authenticates one request. STAT PER REQUEST: a rotated file takes effect on the
// next call, and a file that has become unreadable stops admitting anyone.
func (s *server) admits(w http.ResponseWriter, r *http.Request) bool {
	if err := s.reloadTokens(); err != nil {
		refuse(w, http.StatusServiceUnavailable, "media.credential_unreadable",
			"this server cannot read its own token-hash file, so it admits nobody: "+err.Error(),
			"the pod's provisioner owns that file; a media server never mints a credential")
		return false
	}
	presented := strings.TrimSpace(strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer "))
	if presented == "" {
		refuse(w, http.StatusUnauthorized, "media.unauthenticated",
			"this route takes a bearer credential and none was presented",
			"the rental's provisioned owner token is the bearer")
		return false
	}
	s.mu.Lock()
	hashes := append([]string(nil), s.hashes...)
	s.mu.Unlock()
	for _, line := range hashes {
		if secret.MatchesHash(presented, line) {
			return true
		}
	}
	refuse(w, http.StatusUnauthorized, "media.unauthenticated",
		"the presented credential is not one this pod was provisioned with",
		"`cozy rent ls` renders the digest of the token this host holds")
	return false
}

const maxBootstrapReceiptBytes = 64 << 10

func validTokenHash(line string) bool {
	if !strings.HasPrefix(line, "sha256:") || len(line) != len("sha256:")+64 {
		return false
	}
	_, err := hex.DecodeString(strings.TrimPrefix(line, "sha256:"))
	return err == nil && line == strings.ToLower(line)
}

// bootstrapReceipt is Tensorhub's one read-only rendezvous with a pod it bought. The
// attempt-specific hash is separate from the renter set, so the hub can prove that the
// expected image answered without acquiring the ability to control the worker or move
// media. The file may appear after this listener starts; until then readiness is pending.
// Exact bytes are returned so the hub can digest and retain the document it actually saw.
func (s *server) bootstrapReceipt(w http.ResponseWriter, r *http.Request) {
	presented := strings.TrimSpace(strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer "))
	if !secret.MatchesHash(presented, s.opt.bootstrapTokenHash) {
		refuse(w, http.StatusUnauthorized, "media.bootstrap_unauthenticated",
			"the presented credential does not match this acquisition attempt",
			"Tensorhub alone holds the one-attempt bootstrap credential")
		return
	}
	file, err := os.Open(s.opt.bootstrapReceipt)
	if err != nil {
		refuse(w, http.StatusServiceUnavailable, "media.bootstrap_pending",
			"the pod has not published its readiness receipt: "+err.Error(),
			"wait for endpoint materialization and both pod listeners")
		return
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, maxBootstrapReceiptBytes+1))
	if err != nil || len(data) == 0 || len(data) > maxBootstrapReceiptBytes || !json.Valid(data) {
		refuse(w, http.StatusServiceUnavailable, "media.bootstrap_receipt_invalid",
			"the pod's readiness receipt is unreadable, empty, oversized, or not JSON",
			"publish one complete receipt by atomic rename after endpoint materialization")
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Length", strconv.Itoa(len(data)))
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(data)
}

// safeName is the only shape a caller-supplied path segment may take. An output id is a
// declared RESULT FIELD PATH (`image`, `detail.thumb`) and a slot or blob id is opaque
// hex, so the set is small and closed on purpose: `..` and every separator are outside it,
// which makes traversal unrepresentable rather than filtered.
var safeName = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`)

func named(w http.ResponseWriter, r *http.Request, keys ...string) ([]string, bool) {
	out := make([]string, 0, len(keys))
	for _, key := range keys {
		v := r.PathValue(key)
		if !safeName.MatchString(v) || strings.Contains(v, "..") {
			refuse(w, http.StatusBadRequest, "media.bad_name",
				fmt.Sprintf("%q is not a usable %s", v, key),
				"a name here is letters, digits, dot, dash and underscore — a separator or "+
					"a parent reference is not a name this server can hold")
			return nil, false
		}
		out = append(out, v)
	}
	return out, true
}

// used is the ONE admission number: bytes already written outside active output slots,
// plus every active slot's exact grant bound. Worker-written files under a reserved slot
// are excluded because their maximum has already been charged in full. That makes this
// number invariant while the other process writes and closes the cross-process check/write
// race without teaching the worker a second quota authority or relying on headroom.
//
// Binding records are included even though the worker resolves them outside this subtree.
// A malformed reservation fails closed by reporting the whole quota used.
func (s *server) used() int64 {
	reservations, err := s.reservations()
	if err != nil {
		return s.opt.quota
	}
	var total int64
	count := func(path string, info os.FileInfo, err error) error {
		if err == nil && info != nil && !info.IsDir() {
			rel, relErr := filepath.Rel(s.opt.root, path)
			if relErr == nil {
				parts := strings.Split(rel, string(filepath.Separator))
				if len(parts) >= 3 && parts[0] == "outputs" {
					if _, reserved := reservations[parts[1]]; reserved {
						return nil
					}
				}
			}
			total += info.Size()
		}
		return nil
	}
	reservationDir := filepath.Join(s.opt.root, ".reservations")
	_ = filepath.Walk(s.opt.root, func(path string, info os.FileInfo, err error) error {
		if err == nil && info != nil && info.IsDir() && path == reservationDir {
			return filepath.SkipDir
		}
		return count(path, info, err)
	})
	if s.opt.plans != "" {
		_ = filepath.Walk(s.opt.plans, count)
	}
	for _, bytes := range reservations {
		total += bytes
	}
	return total
}

func (s *server) reservations() (map[string]int64, error) {
	entries, err := os.ReadDir(filepath.Join(s.opt.root, ".reservations"))
	if os.IsNotExist(err) {
		return map[string]int64{}, nil
	}
	if err != nil {
		return nil, err
	}
	out := make(map[string]int64, len(entries))
	for _, entry := range entries {
		if entry.IsDir() || !safeName.MatchString(entry.Name()) || strings.Contains(entry.Name(), "..") {
			return nil, fmt.Errorf("invalid media reservation %q", entry.Name())
		}
		data, err := os.ReadFile(filepath.Join(s.opt.root, ".reservations", entry.Name()))
		if err != nil {
			return nil, err
		}
		bytes, err := strconv.ParseInt(strings.TrimSpace(string(data)), 10, 64)
		if err != nil || bytes < 0 {
			return nil, fmt.Errorf("invalid byte bound in media reservation %q", entry.Name())
		}
		out[entry.Name()] = bytes
	}
	return out, nil
}

// take reads one request body under two bounds: this server's per-body cap, and whatever
// room the quota leaves. The DECLARED length refuses first when the caller declared one,
// so an oversized upload is answered before its bytes move.
func (s *server) take(w http.ResponseWriter, r *http.Request) ([]byte, bool) {
	room := s.opt.quota - s.used()
	limit := s.opt.maxBody
	if room < limit {
		limit = room
	}
	if limit <= 0 {
		refuse(w, http.StatusInsufficientStorage, "media.quota_exhausted",
			fmt.Sprintf("this pod's media subtree holds %d B of its %d B quota", s.used(), s.opt.quota),
			"outputs already downloaded can be dropped; a media subtree is deliberately "+
				"separate from the worker's journal so a full one never stops an attempt")
		return nil, false
	}
	if r.ContentLength > limit {
		refuse(w, http.StatusRequestEntityTooLarge, "media.over_bound",
			fmt.Sprintf("the body declares %d B and %d B is admissible here", r.ContentLength, limit),
			"the DECLARED length refuses before a byte moves; the bytes meet the same bound")
		return nil, false
	}
	data, err := io.ReadAll(io.LimitReader(r.Body, limit+1))
	if err != nil {
		refuse(w, http.StatusBadRequest, "media.body_unreadable",
			"the request body ended early: "+err.Error(), "re-send it")
		return nil, false
	}
	if int64(len(data)) > limit {
		refuse(w, http.StatusRequestEntityTooLarge, "media.over_bound",
			fmt.Sprintf("the body passed %d B and was cut there", limit),
			"a stream may not exceed what this server admits, and finding out afterwards "+
				"is not a bound")
		return nil, false
	}
	return data, true
}

// commit writes bytes atomically: a temporary beside the destination, then a rename. A
// reader of this subtree therefore never sees a half-written object, which matters because
// the other reader is a worker that does not coordinate with this process at all.
func commit(path string, data []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	staging := filepath.Join(filepath.Dir(path), "."+filepath.Base(path)+".staging")
	defer os.Remove(staging)
	if err := os.WriteFile(staging, data, 0o644); err != nil {
		return err
	}
	return os.Rename(staging, path)
}

func digestOf(data []byte) string {
	sum := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(sum[:])
}

// putInput lands one attempt input and answers the POD-LOCAL PATH. The path in the answer
// is the whole point of the route: the owner mints DeliveryGrant access out of it and has
// no other way to know where the pod's disk is.
func (s *server) putInput(w http.ResponseWriter, r *http.Request) {
	if !s.admits(w, r) {
		return
	}
	names, ok := named(w, r, "blob")
	if !ok {
		return
	}
	s.writes.Lock()
	defer s.writes.Unlock()
	data, ok := s.take(w, r)
	if !ok {
		return
	}
	path := filepath.Join(s.opt.root, "inputs", names[0])
	if err := commit(path, data); err != nil {
		refuse(w, http.StatusInternalServerError, "media.unwritable",
			"the input could not be landed: "+err.Error(), "check the pod's media subtree")
		return
	}
	answer(w, http.StatusCreated, map[string]any{
		"path": path, "digest": digestOf(data), "length": len(data),
	})
}

// putPlan delivers one binding-plan record into the worker's own directory, and is the
// fail-closed half of remote plan delivery (#506a/#506b).
//
// The record is not taken on trust: its IDENTITY is re-hashed here and must equal the id
// it was delivered under. A record that does not hash to its own name is not the plan the
// orchestrator is about to name in a directive, whoever sent it — so it is refused rather
// than staged, and the worker never resolves a plan id against bytes nobody agreed to.
func (s *server) putPlan(w http.ResponseWriter, r *http.Request) {
	if !s.admits(w, r) {
		return
	}
	if s.opt.plans == "" {
		refuse(w, http.StatusNotFound, "media.no_plan_plane",
			"this media server was provisioned with no binding-plan directory",
			"the pod's provisioner passes --plans <worker home>/binding-plans")
		return
	}
	names, ok := named(w, r, "id")
	if !ok {
		return
	}
	s.writes.Lock()
	defer s.writes.Unlock()
	data, ok := s.take(w, r)
	if !ok {
		return
	}
	// The wire spelling is the file spelling: bare hex, which is what the runtime resolves
	// a wire plan id against on its own disk.
	claimed := "sha256:" + strings.TrimSuffix(names[0], ".json")
	if _, e := plan.Verify(data, claimed); e != nil {
		refuse(w, http.StatusBadRequest, "media."+e.ErrName(), e.Message, e.Remedy)
		return
	}
	path := filepath.Join(s.opt.plans, plan.FileName(claimed))
	if err := commit(path, data); err != nil {
		refuse(w, http.StatusInternalServerError, "media.unwritable",
			"the binding record could not be staged: "+err.Error(), "check the pod's worker home")
		return
	}
	answer(w, http.StatusCreated, map[string]any{
		"path": path, "plan_id": claimed, "length": len(data),
	})
}

// reserveOutputs charges one attempt's exact maximum output bytes BEFORE returning its
// pod-local directory. The durable reservation is the protocol between this process and
// the filesystem-writing worker: HTTP admission counts the bound instead of racing the
// worker's current file sizes. A retry with the same bound is idempotent; changing it is
// a conflict because an already-issued grant cannot be widened in place.
func (s *server) reserveOutputs(w http.ResponseWriter, r *http.Request) {
	if !s.admits(w, r) {
		return
	}
	names, ok := named(w, r, "slot")
	if !ok {
		return
	}
	s.writes.Lock()
	defer s.writes.Unlock()
	bound, err := strconv.ParseInt(r.URL.Query().Get("max_bytes"), 10, 64)
	if err != nil || bound < 0 {
		refuse(w, http.StatusBadRequest, "media.bad_reservation",
			"max_bytes must be the non-negative sum of this attempt's output grants",
			"the owner derives it from the OutputBinding max_bytes values")
		return
	}
	slot := names[0]
	reservation := filepath.Join(s.opt.root, ".reservations", slot)
	created := false
	if data, readErr := os.ReadFile(reservation); readErr == nil {
		prior, parseErr := strconv.ParseInt(strings.TrimSpace(string(data)), 10, 64)
		if parseErr != nil || prior < 0 {
			refuse(w, http.StatusInsufficientStorage, "media.reservation_corrupt",
				"the existing output reservation is unreadable, so quota admission fails closed",
				"drop the attempt or repair the pod's media subtree")
			return
		}
		if prior != bound {
			refuse(w, http.StatusConflict, "media.reservation_conflict",
				fmt.Sprintf("slot %s already reserves %d B, not %d B", slot, prior, bound),
				"an issued output grant is immutable; use a new attempt slot")
			return
		}
	} else if !os.IsNotExist(readErr) {
		refuse(w, http.StatusInsufficientStorage, "media.reservation_unreadable",
			"the output reservation cannot be read: "+readErr.Error(),
			"check the pod's media subtree")
		return
	} else {
		used := s.used()
		if bound > s.opt.quota-used {
			refuse(w, http.StatusInsufficientStorage, "media.quota_exhausted",
				fmt.Sprintf("this pod holds or reserves %d B; %d B of its %d B quota remains", used, max(0, s.opt.quota-used), s.opt.quota),
				"drop completed attempts before reserving another output grant")
			return
		}
		if err := commit(reservation, []byte(strconv.FormatInt(bound, 10))); err != nil {
			refuse(w, http.StatusInternalServerError, "media.unwritable",
				"the output reservation could not be recorded: "+err.Error(), "check the pod's media subtree")
			return
		}
		created = true
	}
	dir := filepath.Join(s.opt.root, "outputs", slot)
	if err := os.MkdirAll(dir, 0o777); err != nil {
		if created {
			_ = os.Remove(reservation)
		}
		refuse(w, http.StatusInternalServerError, "media.unwritable",
			"the output slot could not be reserved: "+err.Error(), "check the pod's media subtree")
		return
	}
	// The WORKER writes here and this server only reads: the directory is group/other
	// writable on purpose, because the two processes share a container and not a uid in
	// every provisioning. The subtree is quota-bounded and holds no credential.
	_ = os.Chmod(dir, 0o777)
	answer(w, http.StatusCreated, map[string]any{"dir": dir, "slot": slot})
}

// dropAttempt removes only one opaque attempt slot. Inputs use the slot as an exact
// prefix followed by a dash; outputs use it as their directory name. Plans are shared by
// placements and deliberately outside this ownership boundary.
func (s *server) dropAttempt(w http.ResponseWriter, r *http.Request) {
	if !s.admits(w, r) {
		return
	}
	names, ok := named(w, r, "slot")
	if !ok {
		return
	}
	s.writes.Lock()
	defer s.writes.Unlock()
	slot := names[0]
	entries, err := os.ReadDir(filepath.Join(s.opt.root, "inputs"))
	if err != nil && !os.IsNotExist(err) {
		refuse(w, http.StatusInternalServerError, "media.unreadable",
			"the attempt input directory cannot be read: "+err.Error(), "check the pod's media subtree")
		return
	}
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), slot+"-") {
			if err := os.Remove(filepath.Join(s.opt.root, "inputs", entry.Name())); err != nil && !os.IsNotExist(err) {
				refuse(w, http.StatusInternalServerError, "media.unwritable",
					"an attempt input could not be removed: "+err.Error(), "check the pod's media subtree")
				return
			}
		}
	}
	if err := os.RemoveAll(filepath.Join(s.opt.root, "outputs", slot)); err != nil {
		refuse(w, http.StatusInternalServerError, "media.unwritable",
			"the attempt output directory could not be removed: "+err.Error(), "check the pod's media subtree")
		return
	}
	if err := os.Remove(filepath.Join(s.opt.root, ".reservations", slot)); err != nil && !os.IsNotExist(err) {
		refuse(w, http.StatusInternalServerError, "media.unwritable",
			"the attempt output reservation could not be removed: "+err.Error(), "check the pod's media subtree")
		return
	}
	answer(w, http.StatusOK, map[string]any{"slot": slot})
}

// getOutput serves one committed output. READ-ONLY, and that is what makes this route
// unable to affect an attempt: it opens a file the worker wrote through the filesystem and
// has no way to create, move or remove one.
func (s *server) getOutput(w http.ResponseWriter, r *http.Request) {
	if !s.admits(w, r) {
		return
	}
	names, ok := named(w, r, "slot", "name")
	if !ok {
		return
	}
	path := filepath.Join(s.opt.root, "outputs", names[0], names[1])
	// BOUNDED, even though the worker wrote it. The slot reservation is an aggregate bound,
	// not permission to load any one object into memory, so this read is held to the
	// server's per-body cap as well. An oversized output is refused rather than loaded.
	if info, err := os.Stat(path); err == nil && info.Size() > s.opt.maxBody {
		refuse(w, http.StatusRequestEntityTooLarge, "media.over_bound",
			fmt.Sprintf("the output at %s/%s is %d B and %d B is admissible here",
				names[0], names[1], info.Size(), s.opt.maxBody),
			"raise --max-body on this pod's media server, or grant a smaller output")
		return
	}
	data, err := os.ReadFile(path)
	if err != nil {
		refuse(w, http.StatusNotFound, "media.absent",
			fmt.Sprintf("no output %q in slot %q on this pod", names[1], names[0]),
			"an attempt that has not written its outputs yet has none to fetch, and an "+
				"attempt that failed to write is a terminal the owner must not ack")
		return
	}
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("X-Cozy-Digest", digestOf(data))
	w.Header().Set("Content-Length", strconv.Itoa(len(data)))
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(data)
}

// health answers that this server is up and says what it is holding. It is authenticated
// like everything else: an unauthenticated liveness route would be a second, weaker door.
func (s *server) health(w http.ResponseWriter, r *http.Request) {
	if !s.admits(w, r) {
		return
	}
	answer(w, http.StatusOK, map[string]any{
		"media": "cozy.media/1", "root": s.opt.root,
		"used_bytes": s.used(), "quota_bytes": s.opt.quota,
		"plans": s.opt.plans != "",
	})
}

func (s *server) routes() *http.ServeMux {
	mux := http.NewServeMux()
	if s.opt.bootstrapReceipt != "" {
		mux.HandleFunc("GET /v1/bootstrap/receipt", s.bootstrapReceipt)
	}
	mux.HandleFunc("GET /v1/health", s.health)
	mux.HandleFunc("PUT /v1/inputs/{blob}", s.putInput)
	mux.HandleFunc("PUT /v1/plans/{id}", s.putPlan)
	mux.HandleFunc("POST /v1/outputs/{slot}", s.reserveOutputs)
	mux.HandleFunc("GET /v1/outputs/{slot}/{name}", s.getOutput)
	mux.HandleFunc("DELETE /v1/attempts/{slot}", s.dropAttempt)
	return mux
}

func answer(w http.ResponseWriter, status int, body map[string]any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}

// refuse renders a TYPED refusal, the same envelope shape every other cozy surface uses:
// a code a caller can branch on, a message, and the remedy.
func refuse(w http.ResponseWriter, status int, code, message, remedy string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]string{
		"code": code, "message": message, "remedy": remedy,
	}})
}

func usage(format string, args ...any) int {
	fmt.Fprintf(os.Stderr, "cozy-media: "+format+"\n", args...)
	fmt.Fprintln(os.Stderr, "usage: cozy-media --listen <host:port> --root <dir> "+
		"--tokens <file> [--plans <dir>] [--out <dir>] [--tls-cert <pem> --tls-key <pem>] "+
		"[--bootstrap-receipt <json> --bootstrap-token-sha256 <sha256:hex>] "+
		"[--quota <bytes>] [--max-body <bytes>]")
	return 2
}

func fatal(format string, args ...any) int {
	fmt.Fprintf(os.Stderr, "cozy-media: "+format+"\n", args...)
	return 1
}
