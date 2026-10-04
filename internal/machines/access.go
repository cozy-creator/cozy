package machines

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/capability"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/flock"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/installkey"
	"github.com/cozy-creator/cozy/internal/machinev1"
	"github.com/cozy-creator/cozy/internal/workertls"
)

const executionAccessFile = "execution-access.json"
const executionAccessResets = "execution-access-resets.json"

type executionAccess struct {
	Origin      string            `json:"origin"`
	Token       string            `json:"token"`
	ExpiresAt   int64             `json:"expires_at"`
	Environment map[string]string `json:"environment"`
	CA          string            `json:"ca_der_b64url,omitempty"`
	Certificate string            `json:"certificate_sha256"`
	Credential  string            `json:"credential_identity"`
	Generation  uint64            `json:"generation,omitempty"`
}

type accessReset struct {
	AgentOrigin string `json:"agent_origin"`
	Generation  uint64 `json:"generation"`
	Pending     bool   `json:"pending"`
}

type accessState struct {
	cache  map[string]json.RawMessage
	resets map[string]accessReset
}

func accessOrigin(value string) string {
	u, err := url.Parse(value)
	if err != nil || u.Host == "" {
		return value
	}
	port := u.Port()
	if port == "" {
		port = "443"
		if u.Scheme == "http" {
			port = "80"
		}
	} else if number, err := strconv.Atoi(port); err == nil {
		port = strconv.Itoa(number)
	}
	return strings.ToLower(u.Scheme) + "://" + net.JoinHostPort(strings.ToLower(u.Hostname()), port)
}

// accessCache is this controller's record of one machine's delegated Hub access: the
// grants it attached, per login origin, and the removals it still owes the machine.
type accessCache struct{ dir string }

// accessTarget is the pinned machine a cache's signed access calls reach. A machine on
// this computer reads a loopback login Hub at loopback; any other reads its declared origin.
type accessTarget struct {
	addr, worker string
	local        bool
	leaf         []byte
	pin          func() (*workertls.Pin, *exit.Error)
	owner        func() (installkey.Key, *exit.Error)
}

func (c accessCache) path(name string) string { return filepath.Join(c.dir, name) }

func (c accessCache) lock(ctx context.Context, name string, wait bool) (func(), *exit.Error) {
	if err := os.MkdirAll(c.dir, 0700); err != nil {
		return nil, exit.Internalf("cannot open scoped access storage: %s", err)
	}
	file, err := os.OpenFile(c.path(name), os.O_CREATE|os.O_RDWR, 0600)
	if err != nil {
		return nil, exit.Internalf("cannot open scoped access lock: %s", err)
	}
	err = flock.Exclusive(file)
	if err != nil && wait {
		err = flock.Wait(ctx, file)
	}
	if err != nil {
		file.Close()
		return nil, exit.Named(exit.Unavailable, "machine.access_busy", "another scoped access operation is in progress")
	}
	return func() { _ = flock.Release(file); _ = file.Close() }, nil
}

// The cache lock protects only short local I/O. Logout can erase credentials and
// queue removal while a network request or a machine installation is blocked.
func (c accessCache) state(ctx context.Context, change func(*accessState) *exit.Error) *exit.Error {
	unlock, problem := c.lock(ctx, "execution-access.lock", true)
	if problem != nil {
		return problem
	}
	defer unlock()
	state := accessState{cache: map[string]json.RawMessage{}, resets: map[string]accessReset{}}
	for _, item := range []struct {
		name  string
		value any
	}{{executionAccessFile, &state.cache}, {executionAccessResets, &state.resets}} {
		raw, err := os.ReadFile(c.path(item.name))
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return exit.Internalf("cannot read scoped access storage: %s", err)
		}
		if len(raw) > 4<<20 || json.Unmarshal(raw, item.value) != nil {
			return exit.Named(exit.Conflict, "machine.access_cache_invalid", "scoped access storage is unreadable")
		}
	}
	if state.cache == nil {
		state.cache = map[string]json.RawMessage{}
	}
	if state.resets == nil {
		state.resets = map[string]accessReset{}
	}
	return change(&state)
}

func (c accessCache) save(state *accessState) *exit.Error {
	// Persist the reset generation before cache mutation, so a crash cannot make an
	// earlier in-flight attachment current again.
	for _, item := range []struct {
		name  string
		value any
	}{{executionAccessResets, state.resets}, {executionAccessFile, state.cache}} {
		if item.name == executionAccessFile && len(state.cache) == 0 {
			if err := os.Remove(c.path(executionAccessFile)); err != nil && !errors.Is(err, os.ErrNotExist) {
				return exit.Internalf("cannot erase scoped access cache: %s", err)
			}
			continue
		}
		raw, err := json.Marshal(item.value)
		if err == nil {
			err = writePrivate(c.path(item.name), raw)
		}
		if err != nil {
			return exit.Internalf("cannot retain scoped access state: %s", err)
		}
	}
	return nil
}

func cachedAccess(state *accessState, origin string) executionAccess {
	for key, raw := range state.cache {
		if accessOrigin(key) == accessOrigin(origin) {
			var grant executionAccess
			_ = json.Unmarshal(raw, &grant)
			return grant
		}
	}
	return executionAccess{}
}
func eraseCachedAccess(state *accessState, origin string) {
	for key := range state.cache {
		if accessOrigin(key) == accessOrigin(origin) {
			delete(state.cache, key)
		}
	}
}

// queueReset forgets origin's cached grant and advances its generation, so an attachment in
// flight discards what it was authorizing. A removal is owed only for a grant this controller
// attached, which it does over cozy.worker.v1 alone: a machine serving cozy.machine.v1 takes
// Hub access inside each run, keeps none, and is never asked to remove any.
func (c accessCache) queueReset(ctx context.Context, origin string) *exit.Error {
	return c.state(ctx, func(state *accessState) *exit.Error {
		key := accessOrigin(origin)
		reset := state.resets[key]
		reset.Generation++
		if attached := cachedAccess(state, origin).Origin; attached != "" {
			reset.AgentOrigin, reset.Pending = attached, true
		}
		state.resets[key] = reset
		eraseCachedAccess(state, origin)
		return c.save(state)
	})
}

func (h *Host) accessCache() accessCache { return accessCache{dir: h.dir} }

// endpointCache is an explicit endpoint's cache, apart from this computer's machine's.
func (h *Host) endpointCache(name string) accessCache {
	return accessCache{dir: filepath.Join(h.dir, "endpoints", name)}
}

func (h *Host) accessTarget(launch *Launch) accessTarget {
	return accessTarget{addr: launch.Addr, worker: launch.WorkerID, local: true, leaf: launch.Leaf, pin: h.Pin, owner: h.Authorize}
}

// ForgetExecutionAccess erases only this login origin's client caches and, where this
// controller attached a grant, durably requests its removal from that machine. It never
// launches a stopped agent; an explicit endpoint's removal runs when the endpoint is next
// reached. pending=true means local erasure succeeded but machine-side cleanup is deferred.
func (h *Host) ForgetExecutionAccess(ctx context.Context, origin string) (pending bool, problem *exit.Error) {
	if problem := h.accessCache().queueReset(ctx, origin); problem != nil {
		return false, problem
	}
	endpoints, _ := os.ReadDir(filepath.Join(h.dir, "endpoints"))
	for _, entry := range endpoints {
		if entry.IsDir() {
			if problem := h.endpointCache(entry.Name()).queueReset(ctx, origin); problem != nil {
				return false, problem
			}
		}
	}
	cache := h.accessCache()
	unlock, problem := cache.lock(ctx, "execution-access-request.lock", false)
	if problem != nil {
		return true, nil
	}
	defer unlock()
	if problem := cache.state(ctx, func(state *accessState) *exit.Error { pending = state.resets[accessOrigin(origin)].Pending; return nil }); problem != nil || !pending {
		return false, problem // nothing was attached: no machine is asked anything
	}
	record, problem := h.record()
	if problem != nil {
		return true, problem
	}
	if record == nil || !h.alive(record.PID) {
		return true, nil
	}
	launch := &Launch{Addr: net.JoinHostPort("127.0.0.1", strconv.Itoa(record.WorkerPort)), WorkerID: record.WorkerID}
	if problem := resetAccessLocked(ctx, cache, h.accessTarget(launch), accessOrigin(origin)); problem != nil {
		return true, problem
	}
	problem = cache.state(ctx, func(state *accessState) *exit.Error { pending = state.resets[accessOrigin(origin)].Pending; return nil })
	return pending, problem
}

// ResumeExecutionAccessCleanup is best-effort on an already-started agent.
func (h *Host) ResumeExecutionAccessCleanup(ctx context.Context, launch *Launch) {
	resumeAccessCleanup(ctx, h.accessCache(), h.accessTarget(launch))
}

// resumeAccessCleanup sends queued removals. A pending unrelated Hub never prevents
// offline work; selected-origin attachment below requires either completed removal or
// a fresh same-principal renewal.
func resumeAccessCleanup(ctx context.Context, cache accessCache, target accessTarget) {
	unlock, problem := cache.lock(ctx, "execution-access-request.lock", false)
	if problem != nil {
		return
	}
	defer unlock()
	var origins []string
	if problem := cache.state(ctx, func(state *accessState) *exit.Error {
		for key, value := range state.resets {
			if value.Pending {
				origins = append(origins, key)
			}
		}
		return nil
	}); problem != nil {
		return
	}
	slices.Sort(origins)
	for _, origin := range origins {
		if resetAccessLocked(ctx, cache, target, origin) != nil {
			return
		}
	}
}

func resetAccessLocked(ctx context.Context, cache accessCache, target accessTarget, origin string) *exit.Error {
	var selected accessReset
	if problem := cache.state(ctx, func(state *accessState) *exit.Error { selected = state.resets[origin]; return nil }); problem != nil {
		return problem
	}
	if !selected.Pending {
		return nil
	}
	code, raw, problem := target.call(ctx, http.MethodDelete, map[string]string{"origin": selected.AgentOrigin})
	if problem != nil {
		return problem
	}
	if code != http.StatusNoContent {
		var refused struct {
			Error struct{ Code string } `json:"error"`
		}
		_ = json.Unmarshal(raw, &refused)
		if code == http.StatusConflict && refused.Error.Code == "machine_busy" {
			return exit.Named(exit.Conflict, "machine.execution_access_busy", "machine-side access removal is queued until accepted work and retained results are finished or released")
		}
		if code == http.StatusNotFound || code == http.StatusMethodNotAllowed {
			return exit.Named(exit.Structural, "machine.agent_update_required", "this machine cannot remove scoped access; update it while idle")
		}
		return exit.Named(exit.Unavailable, "machine.execution_access_cleanup_pending", "machine-side access removal is queued (HTTP %d)", code)
	}
	return cache.state(ctx, func(state *accessState) *exit.Error {
		current := state.resets[origin]
		if current.Generation == selected.Generation {
			current.Pending = false
			state.resets[origin] = current
			return cache.save(state)
		}
		return nil
	})
}

func deviceBoundAccess(token string) bool {
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return false
	}
	header, err := base64.RawURLEncoding.Strict().DecodeString(parts[0])
	if err != nil {
		return false
	}
	var kind struct {
		Type string `json:"typ"`
	}
	if json.Unmarshal(header, &kind) != nil || kind.Type != "delegated-access+jwt" {
		return false
	}
	raw, err := base64.RawURLEncoding.Strict().DecodeString(parts[1])
	if err != nil {
		return false
	}
	var claims struct {
		Permissions []string `json:"permissions"`
		Attributes  struct {
			Device string `json:"execution_device_key_id"`
		} `json:"attributes"`
	}
	return json.Unmarshal(raw, &claims) == nil && claims.Attributes.Device != "" && slices.Equal(claims.Permissions, []string{"cozy.execution-access"})
}

// reads is where the target machine reads the Hub its owner reached at login. A Hub reached
// on loopback runs on this computer, so a machine here reads it there too, never through
// the public origin (a tunnel, a proxy) it declares for remote machines.
func (t accessTarget) reads(login, declared string) string {
	if local := loopbackOrigin(login); t.local && local != "" && loopbackOrigin(declared) == "" {
		return local
	}
	return declared
}

// loopbackOrigin is origin as scheme://host[:port] when it names this computer as every
// scoped-access agent accepts it, else "".
func loopbackOrigin(origin string) string {
	u, err := url.Parse(origin)
	if err != nil || u.Scheme != "http" && u.Scheme != "https" || u.User != nil || u.RawQuery != "" || u.Fragment != "" || u.Path != "" && u.Path != "/" {
		return ""
	}
	switch u.Hostname() {
	case "localhost", "127.0.0.1", "::1":
		return u.Scheme + "://" + u.Host
	}
	return ""
}

func (h *Host) attachAccess(ctx context.Context, launch *Launch, origin string, account *hub.Client) (string, *exit.Error) {
	installed, problem := h.Installed()
	if problem != nil {
		return "", problem
	}
	if installed == nil || !slices.Contains(launch.Capabilities, HubAccessCapability) {
		if servesAPI(ctx, h.binary()) {
			return "", machinev1.NewerThanCozy("it takes Hub access only inside a cozy.machine.v1 run") // never a standing grant
		}
		return "", exit.Named(exit.Structural, "machine.agent_update_required", "delegated Hub access requires %s", HubAccessCapability)
	}
	return attachAccess(ctx, h.accessCache(), h.accessTarget(launch), origin, account)
}

// attachAccess gives the target machine the signed-in account's execution access at
// origin and answers the origin that machine reads. A cached grant is reused while it
// is the same login's, for the same pinned leaf and generation, and far from expiry:
// a run never asks Tensorhub for access the machine already holds.
func attachAccess(ctx context.Context, cache accessCache, target accessTarget, origin string, account *hub.Client) (string, *exit.Error) {
	credential := ""
	if account != nil {
		credential = account.CredentialIdentity()
	}
	if credential == "" {
		return "", exit.Named(exit.Credential, "machine.execution_access_required", "running work from %s requires account-authorized execution access", origin)
	}
	unlock, problem := cache.lock(ctx, "execution-access-request.lock", true)
	if problem != nil {
		return "", problem
	}
	defer unlock()
	key := accessOrigin(origin)
	var grant executionAccess
	snapshot := map[string]accessReset{}
	if problem := cache.state(ctx, func(state *accessState) *exit.Error {
		grant = cachedAccess(state, origin)
		for key, value := range state.resets {
			snapshot[key] = value
		}
		return nil
	}); problem != nil {
		return "", problem
	}
	digest := sha256.Sum256(target.leaf)
	certificate := hex.EncodeToString(digest[:])
	pending := snapshot[key].Pending
	for _, reset := range snapshot {
		if reset.Pending && grant.Origin != "" && accessOrigin(reset.AgentOrigin) == accessOrigin(grant.Origin) {
			pending = true
		}
	}
	refresh := pending || grant.Certificate != certificate || grant.Credential != credential || grant.ExpiresAt <= time.Now().Add(time.Minute).Unix() || !deviceBoundAccess(grant.Token) || grant.Generation != snapshot[key].Generation || accessOrigin(grant.Origin) != accessOrigin(target.reads(origin, grant.Origin))
	if refresh {
		access, problem := account.AuthorizeExecutionAccess(ctx, target.leaf)
		if problem != nil {
			return "", problem
		}
		if !deviceBoundAccess(access.Token) {
			return "", exit.Named(exit.Credential, "hub.execution_access_device_key_required", "Tensorhub must issue execution access bound to the current login device")
		}
		reads := target.reads(origin, access.Environment["TENSORHUB_ORIGIN"])
		access.Environment["TENSORHUB_ORIGIN"] = reads
		grant = executionAccess{Origin: reads, Token: access.Token, ExpiresAt: access.ExpiresAt.Unix(), Environment: access.Environment, Certificate: certificate, Credential: credential, Generation: snapshot[key].Generation}
		if len(access.TrustRoot) > 0 {
			grant.CA = base64.RawURLEncoding.EncodeToString(access.TrustRoot)
		}
	}
	if account.CredentialIdentity() != credential {
		return "", exit.Named(exit.Credential, "machine.execution_account_changed", "the Tensorhub login changed while execution access was authorized")
	}
	for attempt := 0; attempt < 2; attempt++ {
		body := map[string]any{"origin": grant.Origin, "token": grant.Token, "expires_at": grant.ExpiresAt, "environment": grant.Environment, "ca_der_b64url": grant.CA}
		code, raw, problem := target.call(ctx, http.MethodPost, body)
		if problem != nil {
			return "", problem
		}
		var refused struct {
			Error struct{ Code string } `json:"error"`
		}
		_ = json.Unmarshal(raw, &refused)
		if code == http.StatusConflict && refused.Error.Code == "hub_access_principal_conflict" && attempt == 0 {
			if problem := cache.state(ctx, func(state *accessState) *exit.Error {
				for scope, current := range state.resets {
					if (scope == key || accessOrigin(current.AgentOrigin) == accessOrigin(grant.Origin)) && current.Generation != snapshot[scope].Generation {
						return exit.Named(exit.Credential, "machine.execution_access_revoked", "execution access was removed while attachment was in progress; log in again")
					}
				}
				reset := state.resets[key]
				reset.Generation++
				reset.AgentOrigin, reset.Pending = grant.Origin, true
				state.resets[key], snapshot[key] = reset, reset
				grant.Generation = reset.Generation
				eraseCachedAccess(state, origin)
				return cache.save(state)
			}); problem != nil {
				return "", problem
			}
			if problem := resetAccessLocked(ctx, cache, target, key); problem != nil {
				return "", problem
			}
			continue
		}
		if code == http.StatusNotFound || code == http.StatusMethodNotAllowed {
			return "", exit.Named(exit.Structural, "machine.agent_update_required", "this machine cannot accept delegated Hub access; update it")
		}
		if code/100 != 2 {
			return "", exit.Named(exit.Credential, "machine.execution_access_refused", "the machine refused execution access (HTTP %d)", code)
		}
		var reply struct {
			Origin    string `json:"origin"`
			ExpiresAt int64  `json:"expires_at"`
		}
		if json.Unmarshal(raw, &reply) != nil || accessOrigin(reply.Origin) != accessOrigin(grant.Origin) || reply.ExpiresAt != grant.ExpiresAt {
			return "", exit.New(exit.Conflict, "the machine did not acknowledge execution access")
		}
		revoked := false
		problem = cache.state(ctx, func(state *accessState) *exit.Error {
			for other, current := range state.resets {
				if (other == key || accessOrigin(current.AgentOrigin) == accessOrigin(grant.Origin)) && current.Generation != snapshot[other].Generation {
					revoked = true
				}
			}
			if revoked {
				reset := state.resets[key]
				reset.Generation++
				reset.Pending = true
				reset.AgentOrigin = grant.Origin
				state.resets[key] = reset
				eraseCachedAccess(state, origin)
			} else {
				// The agent accepted a fresh grant: either the previous one was
				// absent, or this is a safe same-principal credential renewal.
				// That replaces the revoked bytes even while resumable work exists.
				for scope, reset := range state.resets {
					if scope == key || accessOrigin(reset.AgentOrigin) == accessOrigin(grant.Origin) {
						reset.Pending = false
						state.resets[scope] = reset
					}
				}
				grant.Generation = state.resets[key].Generation
				raw, _ := json.Marshal(grant)
				eraseCachedAccess(state, origin)
				state.cache[origin] = raw
			}
			return cache.save(state)
		})
		if problem != nil {
			return "", problem
		}
		if revoked {
			_ = resetAccessLocked(ctx, cache, target, key)
			return "", exit.Named(exit.Credential, "machine.execution_access_revoked", "execution access was removed while attachment was in progress; log in again")
		}
		return grant.Environment["TENSORHUB_ORIGIN"], nil
	}
	return "", exit.Named(exit.Conflict, "machine.execution_access_principal_conflict", "another account changed the machine's Hub access during attachment")
}

// call sends one owner-signed scoped-access request over the pinned TLS identity.
func (t accessTarget) call(ctx context.Context, method string, body any) (int, []byte, *exit.Error) {
	owner, problem := t.owner()
	if problem != nil {
		return 0, nil, problem
	}
	pin, problem := t.pin()
	if problem != nil {
		return 0, nil, problem
	}
	token, err := capability.MintSigned(owner.Public(), owner.Sign, capability.Grant{Machine: t.worker, Action: "hub-access", Expires: time.Now().Add(5 * time.Minute).Unix()})
	if err != nil {
		return 0, nil, exit.Internalf("cannot authorize machine execution access")
	}
	raw, _ := json.Marshal(body)
	request, err := http.NewRequestWithContext(ctx, method, "https://"+t.addr+"/v1/hubs/access", bytes.NewReader(raw))
	if err != nil {
		return 0, nil, exit.Internalf("cannot address machine access API")
	}
	request.Header.Set("Authorization", "Cozy-Cap "+token)
	request.Header.Set("Content-Type", "application/json")
	transport := &http.Transport{TLSClientConfig: pin.TLSConfig()}
	defer transport.CloseIdleConnections()
	response, err := (&http.Client{Transport: transport, Timeout: 30 * time.Second, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}).Do(request)
	if err != nil {
		return 0, nil, exit.Unavailablef("machine execution access endpoint did not answer")
	}
	defer response.Body.Close()
	raw, err = io.ReadAll(io.LimitReader(response.Body, (64<<10)+1))
	if err != nil || len(raw) > 64<<10 {
		return 0, nil, exit.Unavailablef("machine execution access response was interrupted or too large")
	}
	return response.StatusCode, raw, nil
}
