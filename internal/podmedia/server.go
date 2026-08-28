// Package podmedia is THE POD'S MEDIA PLANE (cl-014, ruled #506b): the byte plane of a
// rented pod, and the only way bytes cross between an owner and the worker it rented.
//
// It is a PACKAGE and not a program (cl-036). It used to be `cmd/cozy-media`, a second
// binary the supervisor exec'd; the pod now ships ONE binary, `cmd/cozy-pod`, and this
// plane runs inside it. The split that remains is the one that carries weight: THIS
// package parses attacker-influenced request bytes and it is fenced against every
// privilege the supervisor holds — it may not exec, signal, or reap, and `cmd/cozy-pod`
// is package main, so nothing here can even NAME the supervision loop. The concentration
// the merge creates (PID 1 co-resident with an HTTP parser) is bought down by that
// boundary rather than by a comment.
//
// It stays coupled to the Python worker BY THE FILESYSTEM ALONE. There is no RPC between
// the two in either direction — the worker never calls this plane and this plane never
// calls the worker — so a crash here loses no attempt state and a killed worker does not
// stop an owner from downloading what it already produced. That is a structural property,
// not a discipline: this package imports no protocol client and, by fence, contains no
// outbound network call of any kind. It joins the liability fence by HAVING NOTHING TO
// EGRESS WITH.
//
// Seven routes, and the shape of each is the whole design:
//
//	GET  /v1/health               the owner's ONE pre-flight: this server names itself and
//	                              its contract revision, and the owner refuses the rental
//	                              rather than upload a byte to a plane at another revision.
//	GET  /v1/bootstrap/receipt    Tensorhub reads one attempt-bound, pod-authored readiness
//	                              envelope. Its HMAC is verified by Tensorhub before the TLS
//	                              peer is trusted; the response is the exact file bytes.
//	PUT  /v1/inputs/{blob}        the owner uploads one attempt input; the answer is the
//	                              POD-LOCAL PATH it landed at, which is what the owner then
//	                              mints into the DeliveryGrant. The owner never guesses a
//	                              pod path and this server never learns an owner path.
//	PUT  /v1/plans/{plan-id}      the owner relays one exact canonical binding plan into the
//	                              worker's own `binding-plans` directory. FAIL-CLOSED: the
//	                              whole byte string is re-hashed HERE and must equal the id
//	                              it was delivered under. Local binding records refuse.
//	POST /v1/outputs/{slot}       the owner reserves one attempt's output directory AND its
//	                              exact maximum byte budget, then is told the pod-local path
//	                              to grant the worker.
//	GET  /v1/outputs/{slot}/{id}  the owner reads back what the worker wrote. READ-ONLY:
//	                              this route opens a file and never creates one.
//	DELETE /v1/attempts/{slot}    after rollback, or after mirror plus outcome ack, the
//	                              owner drops exactly that attempt's inputs and outputs.
//
// Every RENTER route authenticates a bearer against a FROZEN SET OF DIGESTS handed to
// this plane at Bind and never re-read: the renter minted the token, Tensorhub kept only
// its SHA-256, and this process holds that digest and compares digests. It never holds a
// raw token, so there is nothing here to leak. The set is validated once, at Bind, and a
// set that is absent, empty, or malformed refuses to bind rather than serve — an
// unauthenticated media plane is not a degraded mode, and since the supervisor cannot
// reach `supervise` without a bound plane, a bad grant is a POD that does not boot. The
// bootstrap envelope contains no capability; Tensorhub authenticates its exact bytes with
// the attempt HMAC before trusting the TLS peer that served them.
//
// The SUBTREE IS QUOTA-BOUNDED and it is not the worker's root: uploads can never ENOSPC
// the journal. The worker writes outputs through the shared filesystem, so this plane
// durably reserves their full grant bounds before returning a directory. HTTP uploads and
// direct worker writes thereby consume one media-owned budget without a cross-process lock.
package podmedia

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"

	"github.com/cozy-creator/cozy-creator/internal/canonical"
	"github.com/cozy-creator/cozy-creator/internal/mediawire"
	"github.com/cozy-creator/cozy-creator/internal/secret"
)

// Options is the whole launch surface of the media plane. Everything is a path grant or a
// bound; nothing is discovered, nothing is read from the environment, and there is no
// configuration file. `cmd/cozy-pod` fills it from the eight allowlisted pod variables and
// its own fixed image layout — this package never learns either.
type Options struct {
	Listen           string
	Root             string   // the quota-bounded subtree this plane OWNS
	Plans            string   // the worker's `binding-plans` directory — write-only, from here
	TokenHashes      []string // the frozen credential set, `sha256:<64 hex>` each
	Cert             string
	Key              string
	BootstrapReceipt string // exact pod-authored JSON; may appear after the listener starts
	Quota            int64
	MaxBody          int64
}

// DefaultQuota is the media subtree's admitted ceiling when the caller states none.
const DefaultQuota = 8 << 30

// prepare validates the grant and lays out the subtree. FAIL CLOSED: it returns an error
// and no plane, so a malformed grant authenticates NOBODY rather than everybody.
func (opt *Options) prepare() error {
	switch {
	case opt.Listen == "":
		return fmt.Errorf("a listen address is required")
	case opt.Root == "":
		return fmt.Errorf("a root is required: this plane owns exactly one subtree")
	}
	if err := validateTokenHashes(opt.TokenHashes); err != nil {
		return err
	}
	if opt.Quota <= 0 {
		opt.Quota = DefaultQuota
	}
	if opt.MaxBody <= 0 {
		// No private object ceiling sits below the pod's admitted media quota.
		opt.MaxBody = opt.Quota
	}
	for _, dir := range []string{filepath.Join(opt.Root, "inputs"), filepath.Join(opt.Root, "outputs")} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("cannot create %s: %w", dir, err)
		}
	}
	reservationDir := filepath.Join(opt.Root, ".reservations")
	if err := os.MkdirAll(reservationDir, 0o700); err != nil {
		return fmt.Errorf("cannot create the media reservation directory: %w", err)
	}
	// A crash before an atomic reservation rename can leave only its hidden temporary.
	// No grant was returned in that state, so boot removes it before quota is admitted.
	entries, err := os.ReadDir(reservationDir)
	if err != nil {
		return fmt.Errorf("cannot inspect the media reservation directory: %w", err)
	}
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), ".") && strings.HasSuffix(entry.Name(), ".staging") {
			if err := os.Remove(filepath.Join(reservationDir, entry.Name())); err != nil {
				return fmt.Errorf("cannot remove interrupted media reservation %s: %w", entry.Name(), err)
			}
		}
	}
	if opt.Plans != "" {
		if err := os.MkdirAll(opt.Plans, 0o755); err != nil {
			return fmt.Errorf("cannot create the binding-plan directory %s: %w", opt.Plans, err)
		}
	}
	return nil
}

type server struct {
	opt Options

	writes sync.Mutex // quota admission and the filesystem mutation are one critical section
}

// tokenHashPattern is the ONE shape a credential digest may take here: the `sha256:<64
// lowercase hex>` spelling secret.HashLine renders and secret.MatchesHash compares.
var tokenHashPattern = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)

// maxTokenHashes bounds the set. It is the rental's own ceiling (tensorhub mints at most
// this many live hashes per rental), restated as a refusal rather than trusted.
const maxTokenHashes = 16

// validateTokenHashes checks the launch credential set and is the whole of this plane's
// admission policy. FAIL CLOSED: an empty set, a duplicate, an out-of-order entry, an
// over-long set, or anything that is not `sha256:<64 lowercase hex>` yields an error and
// no plane at all, so a malformed grant authenticates NOBODY rather than everybody.
// Sorted and unique is required, not normalized: the caller already owes a canonical set,
// and silently repairing one hides a caller that has drifted.
//
// `cmd/cozy-pod` validates the same digests one more time, in the shape the pod
// environment spells them. That is not redundancy to delete: the plane refuses to bind on
// a set it cannot authenticate, and the merge means a plane that will not bind is a pod
// that does not boot.
func validateTokenHashes(hashes []string) error {
	if len(hashes) == 0 {
		return fmt.Errorf("the media grant takes at least one `sha256:<64 hex>` digest")
	}
	if len(hashes) > maxTokenHashes {
		return fmt.Errorf("the media grant takes at most %d digests, not %d", maxTokenHashes, len(hashes))
	}
	for i, hash := range hashes {
		if !tokenHashPattern.MatchString(hash) {
			return fmt.Errorf("media grant entry %d %q is not `sha256:<64 lowercase hex>`", i+1, hash)
		}
		if i > 0 && hashes[i-1] >= hash {
			return fmt.Errorf("media grant entries must be sorted and unique; %q does not follow %q",
				hash, hashes[i-1])
		}
	}
	return nil
}

// admits authenticates one request against the frozen set. Neither side of the comparison
// is a raw credential: this process holds digests and hashes what was presented.
func (s *server) admits(w http.ResponseWriter, r *http.Request) bool {
	presented := strings.TrimSpace(strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer "))
	if presented == "" {
		refuse(w, http.StatusUnauthorized, "media.unauthenticated",
			"this route takes a bearer credential and none was presented",
			"the rental's provisioned owner token is the bearer")
		return false
	}
	for _, line := range s.opt.TokenHashes {
		if secret.MatchesHash(presented, line) {
			return true
		}
	}
	refuse(w, http.StatusUnauthorized, "media.unauthenticated",
		"the presented credential is not one this pod was provisioned with",
		"`cozy rent ls` renders the digest of the token this host holds")
	return false
}

// bootstrapReceipt is Tensorhub's one read-only rendezvous with a pod it bought. The
// envelope carries an attempt HMAC that Tensorhub verifies over exact payload bytes before
// it trusts the TLS peer. This package owns neither that schema nor its key: it serves the
// atomically published bytes and nothing else. Until the file appears, readiness is early.
func (s *server) bootstrapReceipt(w http.ResponseWriter, r *http.Request) {
	file, err := os.Open(s.opt.BootstrapReceipt)
	if err != nil {
		refuse(w, http.StatusTooEarly, "media.bootstrap_pending",
			"the pod has not published its readiness receipt: "+err.Error(),
			"wait for endpoint materialization and both pod listeners")
		return
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, mediawire.MaxReceiptBytes+1))
	if err != nil || len(data) == 0 || len(data) > mediawire.MaxReceiptBytes || !json.Valid(data) {
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
		return s.opt.Quota
	}
	var total int64
	count := func(path string, info os.FileInfo, err error) error {
		if err == nil && info != nil && !info.IsDir() {
			rel, relErr := filepath.Rel(s.opt.Root, path)
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
	reservationDir := filepath.Join(s.opt.Root, ".reservations")
	_ = filepath.Walk(s.opt.Root, func(path string, info os.FileInfo, err error) error {
		if err == nil && info != nil && info.IsDir() && path == reservationDir {
			return filepath.SkipDir
		}
		return count(path, info, err)
	})
	if s.opt.Plans != "" {
		_ = filepath.Walk(s.opt.Plans, count)
	}
	for _, bytes := range reservations {
		total += bytes
	}
	return total
}

func (s *server) reservations() (map[string]int64, error) {
	entries, err := os.ReadDir(filepath.Join(s.opt.Root, ".reservations"))
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
		data, err := os.ReadFile(filepath.Join(s.opt.Root, ".reservations", entry.Name()))
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
func (s *server) bodyLimit(w http.ResponseWriter, r *http.Request, routeMax int64) (int64, bool) {
	room := s.opt.Quota - s.used()
	limit := s.opt.MaxBody
	if routeMax > 0 && routeMax < limit {
		limit = routeMax
	}
	if room < limit {
		limit = room
	}
	if limit <= 0 {
		refuse(w, http.StatusInsufficientStorage, "media.quota_exhausted",
			fmt.Sprintf("this pod's media subtree holds %d B of its %d B quota", s.used(), s.opt.Quota),
			"outputs already downloaded can be dropped; a media subtree is deliberately "+
				"separate from the worker's journal so a full one never stops an attempt")
		return 0, false
	}
	if r.ContentLength > limit {
		refuse(w, http.StatusRequestEntityTooLarge, "media.over_bound",
			fmt.Sprintf("the body declares %d B and %d B is admissible here", r.ContentLength, limit),
			"the DECLARED length refuses before a byte moves; the bytes meet the same bound")
		return 0, false
	}
	return limit, true
}

func (s *server) take(w http.ResponseWriter, r *http.Request, routeMax int64) ([]byte, bool) {
	limit, ok := s.bodyLimit(w, r, routeMax)
	if !ok {
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

// streamInput atomically lands one bounded upload while hashing/counting it. Encoded
// media never becomes a []byte in the media server.
func (s *server) streamInput(w http.ResponseWriter, r *http.Request,
	destination string) (int64, string, bool) {
	limit, ok := s.bodyLimit(w, r, 0)
	if !ok {
		return 0, "", false
	}
	if err := os.MkdirAll(filepath.Dir(destination), 0o755); err != nil {
		refuse(w, http.StatusInternalServerError, "media.unwritable",
			"the input directory could not be created: "+err.Error(), "check the pod's media subtree")
		return 0, "", false
	}
	staging, err := os.CreateTemp(filepath.Dir(destination), "."+filepath.Base(destination)+".staging-*")
	if err != nil {
		refuse(w, http.StatusInternalServerError, "media.unwritable",
			"the input could not be staged: "+err.Error(), "check the pod's media subtree")
		return 0, "", false
	}
	path := staging.Name()
	keep := false
	defer func() {
		_ = staging.Close()
		if !keep {
			_ = os.Remove(path)
		}
	}()
	hash := sha256.New()
	length, err := io.Copy(io.MultiWriter(staging, hash), io.LimitReader(r.Body, limit+1))
	if err != nil {
		refuse(w, http.StatusBadRequest, "media.body_unreadable",
			"the request body ended early: "+err.Error(), "re-send it")
		return 0, "", false
	}
	if length > limit {
		refuse(w, http.StatusRequestEntityTooLarge, "media.over_bound",
			fmt.Sprintf("the body passed %d B and was cut there", limit),
			"a stream may not exceed what this server admits")
		return 0, "", false
	}
	if r.ContentLength >= 0 && length != r.ContentLength {
		refuse(w, http.StatusBadRequest, "media.body_unreadable",
			fmt.Sprintf("the body declared %d B and delivered %d B", r.ContentLength, length),
			"re-send the exact input")
		return 0, "", false
	}
	if err := staging.Sync(); err != nil {
		refuse(w, http.StatusInternalServerError, "media.unwritable",
			"the staged input could not be synced: "+err.Error(), "check the pod's media subtree")
		return 0, "", false
	}
	if err := staging.Close(); err != nil {
		refuse(w, http.StatusInternalServerError, "media.unwritable",
			"the staged input could not be closed: "+err.Error(), "check the pod's media subtree")
		return 0, "", false
	}
	if err := os.Rename(path, destination); err != nil {
		refuse(w, http.StatusInternalServerError, "media.unwritable",
			"the staged input could not be committed: "+err.Error(), "check the pod's media subtree")
		return 0, "", false
	}
	keep = true
	return length, "sha256:" + hex.EncodeToString(hash.Sum(nil)), true
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
	path := filepath.Join(s.opt.Root, "inputs", names[0])
	length, digest, ok := s.streamInput(w, r, path)
	if !ok {
		return
	}
	answer(w, http.StatusCreated, map[string]any{
		"path": path, "digest": digest, "length": length,
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
	if s.opt.Plans == "" {
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
	data, ok := s.take(w, r, canonical.DocMax)
	if !ok {
		return
	}
	// The wire spelling is the file spelling: bare hex, which is what the runtime resolves
	// a wire plan id against on its own disk. Remote plans are Tensorhub's canonical
	// EntrypointBindingPlan bytes: their subject id is their whole-byte digest. The
	// retired local EntrypointBindingRecord identity is deliberately not accepted here.
	claimed := "sha256:" + strings.TrimSuffix(names[0], ".json")
	doc, err := canonical.ReadObject(data)
	if err != nil || doc.Str("format") != "cozy.endpoint.EntrypointBindingPlan/1" ||
		len(doc) != 4 || doc["bindings"] == nil || doc["descriptor"] == nil || doc["entrypoint"] == nil {
		refuse(w, http.StatusBadRequest, "media.binding_plan_invalid",
			"the delivered bytes are not one exact canonical EntrypointBindingPlan",
			"relay Tensorhub's acquisition-attempt plan bytes; never render a local binding record")
		return
	}
	computed, _ := canonical.Spell(canonical.Digest(data))
	if computed != claimed {
		refuse(w, http.StatusBadRequest, "media.binding_plan_identity_mismatch",
			"the delivered plan hashes to "+computed+", not "+claimed,
			"use the digest and exact canonical bytes from Tensorhub's persisted control snapshot")
		return
	}
	path := filepath.Join(s.opt.Plans, strings.TrimPrefix(claimed, "sha256:")+".json")
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
	reservation := filepath.Join(s.opt.Root, ".reservations", slot)
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
		if bound > s.opt.Quota-used {
			refuse(w, http.StatusInsufficientStorage, "media.quota_exhausted",
				fmt.Sprintf("this pod holds or reserves %d B; %d B of its %d B quota remains", used, max(0, s.opt.Quota-used), s.opt.Quota),
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
	dir := filepath.Join(s.opt.Root, "outputs", slot)
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
	entries, err := os.ReadDir(filepath.Join(s.opt.Root, "inputs"))
	if err != nil && !os.IsNotExist(err) {
		refuse(w, http.StatusInternalServerError, "media.unreadable",
			"the attempt input directory cannot be read: "+err.Error(), "check the pod's media subtree")
		return
	}
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), slot+"-") {
			if err := os.Remove(filepath.Join(s.opt.Root, "inputs", entry.Name())); err != nil && !os.IsNotExist(err) {
				refuse(w, http.StatusInternalServerError, "media.unwritable",
					"an attempt input could not be removed: "+err.Error(), "check the pod's media subtree")
				return
			}
		}
	}
	if err := os.RemoveAll(filepath.Join(s.opt.Root, "outputs", slot)); err != nil {
		refuse(w, http.StatusInternalServerError, "media.unwritable",
			"the attempt output directory could not be removed: "+err.Error(), "check the pod's media subtree")
		return
	}
	if err := os.Remove(filepath.Join(s.opt.Root, ".reservations", slot)); err != nil && !os.IsNotExist(err) {
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
	path := filepath.Join(s.opt.Root, "outputs", names[0], names[1])
	file, err := os.Open(path)
	if err != nil {
		refuse(w, http.StatusNotFound, "media.absent",
			fmt.Sprintf("no output %q in slot %q on this pod", names[1], names[0]),
			"an attempt that has not written its outputs yet has none to fetch, and an "+
				"attempt that failed to write is a terminal the owner must not ack")
		return
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil || !info.Mode().IsRegular() {
		refuse(w, http.StatusNotFound, "media.absent",
			fmt.Sprintf("output %q in slot %q is not one readable regular file", names[1], names[0]),
			"the worker publishes one regular file for each declared output")
		return
	}
	// BOUNDED, even though the worker wrote it. The slot reservation is an aggregate bound,
	// not permission for an oversized object. The owner hashes the stream against the
	// terminal manifest while landing it, so this server does not make a redundant first
	// pass or hold bytes merely to produce a digest header.
	if info.Size() > s.opt.MaxBody {
		refuse(w, http.StatusRequestEntityTooLarge, "media.over_bound",
			fmt.Sprintf("the output at %s/%s is %d B and %d B is admissible here",
				names[0], names[1], info.Size(), s.opt.MaxBody),
			"raise --max-body on this pod's media server, or grant a smaller output")
		return
	}
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("Content-Length", strconv.FormatInt(info.Size(), 10))
	w.WriteHeader(http.StatusOK)
	_, _ = io.CopyN(w, file, info.Size())
}

// health answers that this server is up, WHICH CONTRACT IT SPEAKS, and what it is holding.
// The revision is the whole reason the owner asks before it uploads: this binary is pinned
// into the pod image by commit and the owner floats, so `service` + `contract_rev` is the
// only thing standing between the two ends and a silently misparsed answer. It is
// authenticated like everything else: an unauthenticated liveness route would be a second,
// weaker door.
func (s *server) health(w http.ResponseWriter, r *http.Request) {
	if !s.admits(w, r) {
		return
	}
	answer(w, http.StatusOK, mediawire.Health{
		Contract: mediawire.Ours(), Root: s.opt.Root,
		UsedBytes: s.used(), QuotaBytes: s.opt.Quota, MaxObjectBytes: s.opt.MaxBody,
		Plans: s.opt.Plans != "",
	})
}

func (s *server) routes() *http.ServeMux {
	mux := http.NewServeMux()
	if s.opt.BootstrapReceipt != "" {
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

func answer(w http.ResponseWriter, status int, body any) {
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
