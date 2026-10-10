// Package accountauth owns Creator's machine credential and the one AuthKit
// challenge/sign exchange that turns it into a short-lived Tensorhub bearer.
package accountauth

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/cozy-creator/cozy/internal/capability"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/secret"
)

const (
	credentialVersion = 1
	enrollDomain      = "authkit.device-key-enrollment/1"
	loginDomain       = "authkit.device-key-login/1"
	maxAnswer         = 1 << 20
)

var rawBase64 = base64.RawURLEncoding

type credential struct {
	Version     int    `json:"version"`
	Hub         string `json:"hub"`
	Email       string `json:"email"`
	DeviceKeyID string `json:"device_key_id"`
	PrivateKey  string `json:"private_key"`
}

// Session is the non-secret result of authentication. AccessToken is deliberately
// a secret.Value: only the hub request builder may reveal it into a header.
type Session struct {
	AccessToken secret.Value
	ExpiresAt   time.Time
	DeviceKeyID string
	Email       string
}

// Enrollment is an in-memory transaction. Its private key is never persisted until
// AuthKit has verified both the email code and possession signature.
type Enrollment struct {
	id          string
	challenge   []byte
	private     ed25519.PrivateKey
	email       string
	deviceKeyID string
}

// BeginEmailProof proves current mailbox control for a destructive machine-key
// operation without creating or replacing this machine's key.
func (m *Manager) BeginEmailProof(ctx context.Context) (*Enrollment, time.Time, *exit.Error) {
	stored, private, problem := m.load()
	if problem != nil {
		return nil, time.Time{}, problem
	}
	public := private.Public().(ed25519.PublicKey)
	var begun struct {
		EnrollmentID string `json:"enrollment_id"`
		Challenge    string `json:"challenge"`
		ExpiresAt    string `json:"expires_at"`
	}
	if problem := m.post(ctx, hub.AuthAPI+"/device-keys/enroll/begin", map[string]string{
		"email": stored.Email, "public_key": rawBase64.EncodeToString(public),
	}, &begun); problem != nil {
		return nil, time.Time{}, problem
	}
	challenge, problem := challengeBytes(begun.Challenge)
	if problem != nil || begun.EnrollmentID == "" {
		return nil, time.Time{}, exit.Named(exit.Internal, "auth.unreadable_challenge",
			"Tensorhub returned an invalid email-proof challenge")
	}
	expires, problem := parseExpiry(begun.ExpiresAt)
	if problem != nil {
		return nil, time.Time{}, problem
	}
	return &Enrollment{id: begun.EnrollmentID, challenge: challenge, private: private,
		email: stored.Email, deviceKeyID: stored.DeviceKeyID}, expires, nil
}

// FinishEmailProof returns the short token that proves both the existing
// machine key and current control of its account email.
func (m *Manager) FinishEmailProof(ctx context.Context, enrollment *Enrollment, code string) (Session, *exit.Error) {
	if enrollment == nil || enrollment.deviceKeyID == "" {
		return Session{}, exit.Internalf("email proof was not started")
	}
	code = strings.TrimSpace(code)
	if code == "" {
		return Session{}, exit.Usagef("the email verification code is empty")
	}
	var answer tokenAnswer
	started := m.now()
	if problem := m.post(ctx, hub.AuthAPI+"/device-keys/enroll/finish", map[string]string{
		"enrollment_id": enrollment.id,
		"code":          code,
		"signature":     rawBase64.EncodeToString(sign(enrollment.private, enrollDomain, enrollment.challenge)),
	}, &answer); problem != nil {
		return Session{}, problem
	}
	session, problem := answer.session(enrollment.email, started)
	if problem != nil {
		return Session{}, problem
	}
	if session.DeviceKeyID != enrollment.deviceKeyID {
		return Session{}, exit.Named(exit.Internal, "auth.wrong_machine",
			"Tensorhub returned an email proof for a different machine")
	}
	m.mu.Lock()
	m.session = session
	m.mu.Unlock()
	return session, nil
}

// Forget erases this Hub origin's local machine credential, whatever the server said.
func (m *Manager) Forget() *exit.Error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if err := os.Remove(m.path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return exit.Internalf("cannot erase the machine credential: %s", err)
	}
	_ = os.Remove(m.sessionPath())
	m.session = Session{}
	return nil
}

// Manager owns one Tensorhub origin's machine key and its short access token, kept beside it.
// Obtaining one performs no I/O or network work.
type Manager struct {
	hub     string
	path    string
	http    *http.Client
	mu      sync.Mutex
	session Session
	now     func() time.Time
}

var managers struct {
	sync.Mutex
	byPath map[string]*Manager
}

// New returns this process's one Manager for cfg's Tensorhub origin, so every caller
// addressing that origin shares one short bearer, and a daemon serving several hubs
// holds one credential per origin, each sent only to its own.
func New(cfg config.Config) *Manager {
	origin := strings.TrimRight(strings.TrimSpace(cfg.HubURL), "/")
	sum := sha256.Sum256([]byte(origin))
	path := filepath.Join(cfg.Home, "auth", hex.EncodeToString(sum[:])+".json")
	managers.Lock()
	defer managers.Unlock()
	if manager := managers.byPath[path]; manager != nil {
		return manager
	}
	if managers.byPath == nil {
		managers.byPath = map[string]*Manager{}
	}
	manager := &Manager{hub: origin, path: path, http: hub.HTTP(cfg.HubLiveness), now: time.Now}
	managers.byPath[path] = manager
	return manager
}

// Hub is the one origin this Manager's credential belongs to.
func (m *Manager) Hub() string { return m.hub }

// AccessToken satisfies hub.TokenSource. A daemon reuses the cached token while it is
// comfortably live; a short CLI process performs one cheap login exchange.
func (m *Manager) AccessToken(ctx context.Context) (secret.Value, *exit.Error) {
	session, problem := m.Authenticate(ctx)
	return session.AccessToken, problem
}

// Identity names this origin's machine key without its secret: a login replaces it, so what is
// kept for one key is not read for the next. "" when there is none.
func (m *Manager) Identity() string {
	raw, err := os.ReadFile(m.path)
	var stored credential
	if err != nil || json.Unmarshal(raw, &stored) != nil || stored.DeviceKeyID == "" {
		return ""
	}
	return "key:" + stored.DeviceKeyID
}

// MintCapability signs a grant with this origin's device key, the key every machine the
// account rents holds among its authorized keys.
func (m *Manager) MintCapability(g capability.Grant) (string, *exit.Error) {
	_, private, problem := m.load()
	if problem != nil {
		return "", problem
	}
	token, err := capability.Mint(private, g)
	if err != nil {
		return "", exit.Internalf("cannot sign the capability: %s", err)
	}
	return token, nil
}

// RunCapability is what one run may do at this Hub for the signed-in user (th-241): the
// private operations (RFC 9396 authorization_details) a machine whose leaf thumbprint is
// Workload performs for run Run until Expires. The machine trades it, once, for a token.
type RunCapability struct {
	UserID, Audience, Workload, Run string
	Operations                      []any
	Expires                         time.Time
}

// SignRunCapability signs c with this origin's device key, offline: the compact JWS AuthKit's
// JWT-bearer grant takes inside the machine's assertion.
func (m *Manager) SignRunCapability(c RunCapability) (string, *exit.Error) {
	stored, private, problem := m.load()
	if problem != nil {
		return "", problem
	}
	now := m.now()
	if c.UserID == "" || c.Audience == "" || c.Workload == "" || len(c.Operations) == 0 || !c.Expires.After(now) {
		return "", exit.Internalf("a run capability names its user, Hub, machine, operations and expiry")
	}
	jti := make([]byte, 24)
	if _, err := rand.Read(jti); err != nil {
		return "", exit.Internalf("cannot draw a capability id: %s", err)
	}
	header, err := json.Marshal(map[string]string{"alg": "EdDSA", "typ": "authkit-capability+jwt", "kid": stored.DeviceKeyID})
	if err != nil {
		return "", exit.Internalf("cannot encode the capability: %s", err)
	}
	claims, err := json.Marshal(map[string]any{
		"sub": c.UserID, "aud": c.Audience, "cnf": map[string]string{"jkt": c.Workload}, "run": c.Run,
		"jti": rawBase64.EncodeToString(jti), "iat": now.Unix(), "exp": c.Expires.Unix(),
		"authorization_details": c.Operations,
	})
	if err != nil {
		return "", exit.Internalf("cannot encode the capability: %s", err)
	}
	input := rawBase64.EncodeToString(header) + "." + rawBase64.EncodeToString(claims)
	return input + "." + rawBase64.EncodeToString(ed25519.Sign(private, []byte(input))), nil
}

// CredentialPresent reports whether this Tensorhub origin has a local machine
// record. It does not read or validate secret bytes.
func (m *Manager) CredentialPresent() bool {
	_, err := os.Stat(m.path)
	return !errors.Is(err, os.ErrNotExist)
}

// Authenticate silently proves the persisted machine key and returns a short bearer.
func (m *Manager) Authenticate(ctx context.Context) (Session, *exit.Error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	stored, private, problem := m.load()
	if problem != nil {
		m.session = Session{}
		return Session{}, problem
	}
	// Another CLI process may have signed out or installed a different device key.
	// Cached bearers belong only to the key still present on disk.
	live := func(s Session) bool {
		return s.DeviceKeyID == stored.DeviceKeyID && s.AccessToken.Present() && s.ExpiresAt.After(m.now().Add(30*time.Second))
	}
	if live(m.session) {
		return m.session, nil
	}
	// A bearer another command minted serves this one: each command is a short process.
	if kept := m.loadSession(); live(kept) {
		m.session = kept
		return m.session, nil
	}
	m.session = Session{}
	var begun struct {
		ChallengeID string `json:"challenge_id"`
		Challenge   string `json:"challenge"`
		ExpiresAt   string `json:"expires_at"`
	}
	if problem := m.post(ctx, hub.AuthAPI+"/device-keys/login/begin", map[string]string{
		"device_key_id": stored.DeviceKeyID,
	}, &begun); problem != nil {
		return Session{}, m.keyRefused(problem)
	}
	challenge, problem := challengeBytes(begun.Challenge)
	if problem != nil || begun.ChallengeID == "" {
		return Session{}, exit.Named(exit.Internal, "auth.unreadable_challenge",
			"Tensorhub returned an invalid machine-login challenge")
	}
	signature := sign(private, loginDomain, challenge)
	var answer tokenAnswer
	started := m.now()
	if problem := m.post(ctx, hub.AuthAPI+"/device-keys/login/finish", map[string]string{
		"challenge_id": begun.ChallengeID,
		"signature":    rawBase64.EncodeToString(signature),
	}, &answer); problem != nil {
		return Session{}, m.keyRefused(problem)
	}
	m.session, problem = answer.session(stored.Email, started)
	if problem == nil {
		m.saveSession(m.session)
	}
	return m.session, problem
}

// keptSession is a bearer kept beside its machine key, mode 0600, for the next command.
type keptSession struct {
	AccessToken string    `json:"access_token"`
	ExpiresAt   time.Time `json:"expires_at"`
	DeviceKeyID string    `json:"device_key_id"`
	Email       string    `json:"email"`
}

func (m *Manager) sessionPath() string { return strings.TrimSuffix(m.path, ".json") + ".session.json" }

func (m *Manager) loadSession() Session {
	raw, err := os.ReadFile(m.sessionPath())
	var kept keptSession
	if err != nil || json.Unmarshal(raw, &kept) != nil {
		return Session{}
	}
	return Session{AccessToken: secret.New(kept.AccessToken), ExpiresAt: kept.ExpiresAt, DeviceKeyID: kept.DeviceKeyID, Email: kept.Email}
}

// saveSession is best effort: a bearer that cannot be kept is minted again next time.
func (m *Manager) saveSession(s Session) {
	raw, err := json.Marshal(keptSession{AccessToken: s.AccessToken.Reveal(), ExpiresAt: s.ExpiresAt, DeviceKeyID: s.DeviceKeyID, Email: s.Email})
	if err != nil {
		return
	}
	temporary, err := os.CreateTemp(filepath.Dir(m.path), ".session-*.tmp")
	if err != nil {
		return
	}
	defer os.Remove(temporary.Name())
	if err = temporary.Chmod(0o600); err == nil {
		_, err = temporary.Write(raw)
	}
	if closeErr := temporary.Close(); err == nil && closeErr == nil {
		_ = os.Rename(temporary.Name(), m.sessionPath())
	}
}

// BeginEnrollment sends the email code and retains a fresh key only in memory.
func (m *Manager) BeginEnrollment(ctx context.Context, email string) (*Enrollment, time.Time, *exit.Error) {
	email = strings.TrimSpace(email)
	if email == "" {
		return nil, time.Time{}, exit.Usagef("email is required")
	}
	public, private, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return nil, time.Time{}, exit.Internalf("cannot generate the machine key: %s", err)
	}
	var begun struct {
		EnrollmentID string `json:"enrollment_id"`
		Challenge    string `json:"challenge"`
		ExpiresAt    string `json:"expires_at"`
	}
	if problem := m.post(ctx, hub.AuthAPI+"/device-keys/enroll/begin", map[string]string{
		"email": email, "public_key": rawBase64.EncodeToString(public),
	}, &begun); problem != nil {
		return nil, time.Time{}, problem
	}
	challenge, problem := challengeBytes(begun.Challenge)
	if problem != nil || begun.EnrollmentID == "" {
		return nil, time.Time{}, exit.Named(exit.Internal, "auth.unreadable_challenge",
			"Tensorhub returned an invalid machine-enrollment challenge")
	}
	expires, problem := parseExpiry(begun.ExpiresAt)
	if problem != nil {
		return nil, time.Time{}, problem
	}
	return &Enrollment{id: begun.EnrollmentID, challenge: challenge, private: private, email: email}, expires, nil
}

// FinishEnrollment proves the emailed code and the pending private key, then commits
// the machine credential atomically. A failed enrollment leaves no private key behind.
func (m *Manager) FinishEnrollment(ctx context.Context, enrollment *Enrollment, code string) (Session, *exit.Error) {
	if enrollment == nil || enrollment.id == "" {
		return Session{}, exit.Internalf("machine enrollment was not started")
	}
	code = strings.TrimSpace(code)
	if code == "" {
		return Session{}, exit.Usagef("the email verification code is empty")
	}
	var answer tokenAnswer
	started := m.now()
	if problem := m.post(ctx, hub.AuthAPI+"/device-keys/enroll/finish", map[string]string{
		"enrollment_id": enrollment.id,
		"code":          code,
		"signature":     rawBase64.EncodeToString(sign(enrollment.private, enrollDomain, enrollment.challenge)),
	}, &answer); problem != nil {
		return Session{}, problem
	}
	session, problem := answer.session(enrollment.email, started)
	if problem != nil {
		return Session{}, problem
	}
	record := credential{
		Version: credentialVersion, Hub: m.hub, Email: enrollment.email,
		DeviceKeyID: session.DeviceKeyID,
		PrivateKey:  rawBase64.EncodeToString(enrollment.private.Seed()),
	}
	if problem := m.save(record); problem != nil {
		return Session{}, problem
	}
	m.mu.Lock()
	m.session = session
	m.saveSession(session)
	m.mu.Unlock()
	return session, nil
}

type tokenAnswer struct {
	TokenSet struct {
		AccessToken string `json:"access_token"`
		TokenType   string `json:"token_type"`
		ExpiresIn   int64  `json:"expires_in"`
	} `json:"token_set"`
	DeviceKey struct {
		ID        string `json:"id"`
		Label     string `json:"label"`
		CreatedAt string `json:"created_at"`
	} `json:"device_key"`
}

func (a tokenAnswer) session(email string, started time.Time) (Session, *exit.Error) {
	if a.TokenSet.ExpiresIn <= 0 || a.TokenSet.ExpiresIn > int64((1<<63-1)/time.Second) {
		return Session{}, exit.Named(exit.Internal, "auth.unreadable_expiry",
			"Tensorhub returned an invalid authentication expiry")
	}
	if a.TokenSet.TokenType != "Bearer" || strings.TrimSpace(a.TokenSet.AccessToken) == "" || strings.TrimSpace(a.DeviceKey.ID) == "" {
		return Session{}, exit.Named(exit.Internal, "auth.unreadable_token",
			"Tensorhub returned an invalid machine access token")
	}
	// Start before the exchange so network latency cannot extend the server's TTL.
	return Session{AccessToken: secret.New(a.TokenSet.AccessToken),
		ExpiresAt:   started.Add(time.Duration(a.TokenSet.ExpiresIn) * time.Second),
		DeviceKeyID: a.DeviceKey.ID, Email: email}, nil
}

func sign(private ed25519.PrivateKey, domain string, challenge []byte) []byte {
	message := make([]byte, 0, len(domain)+1+len(challenge))
	message = append(message, domain...)
	message = append(message, 0)
	message = append(message, challenge...)
	return ed25519.Sign(private, message)
}

func challengeBytes(encoded string) ([]byte, *exit.Error) {
	decoded, err := rawBase64.DecodeString(encoded)
	if err != nil || len(decoded) != 32 {
		return nil, exit.Named(exit.Internal, "auth.unreadable_challenge",
			"Tensorhub returned a malformed machine challenge")
	}
	return decoded, nil
}

func parseExpiry(value string) (time.Time, *exit.Error) {
	parsed, err := time.Parse(time.RFC3339Nano, value)
	if err != nil || parsed.IsZero() {
		return time.Time{}, exit.Named(exit.Internal, "auth.unreadable_expiry",
			"Tensorhub returned an invalid authentication expiry")
	}
	return parsed, nil
}

func (m *Manager) load() (credential, ed25519.PrivateKey, *exit.Error) {
	info, err := os.Stat(m.path)
	if errors.Is(err, os.ErrNotExist) {
		return credential{}, nil, exit.Named(exit.Credential, "auth.machine_key_missing",
			"this machine is not logged in to %s", m.hub).
			WithNext(m.loginCommand())
	}
	if err != nil {
		return credential{}, nil, exit.Named(exit.Credential, "auth.machine_key_unreadable",
			"the machine credential cannot be read: %s", err).
			WithNext(m.loginCommand())
	}
	if runtime.GOOS != "windows" && info.Mode().Perm()&0o077 != 0 {
		return credential{}, nil, exit.Named(exit.Credential, "auth.machine_key_permissions",
			"the machine credential is mode %#o, not 0600", info.Mode().Perm()).
			WithRemedy("restrict the file to its owner, then retry")
	}
	data, err := os.ReadFile(m.path)
	if err != nil {
		return credential{}, nil, exit.Named(exit.Credential, "auth.machine_key_unreadable",
			"the machine credential cannot be read: %s", err)
	}
	var stored credential
	if err := json.Unmarshal(data, &stored); err != nil {
		return credential{}, nil, exit.Named(exit.Credential, "auth.machine_key_invalid",
			"the stored machine credential is invalid").WithNext(m.loginCommand())
	}
	seed, err := rawBase64.DecodeString(stored.PrivateKey)
	if stored.Version != credentialVersion || stored.Hub != m.hub || stored.DeviceKeyID == "" ||
		err != nil || len(seed) != ed25519.SeedSize {
		return credential{}, nil, exit.Named(exit.Credential, "auth.machine_key_invalid",
			"the stored machine credential is invalid").WithNext(m.loginCommand())
	}
	return stored, ed25519.NewKeyFromSeed(seed), nil
}

func (m *Manager) save(stored credential) *exit.Error {
	dir := filepath.Dir(m.path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return exit.Internalf("cannot create the private authentication directory: %s", err)
	}
	if err := os.Chmod(dir, 0o700); err != nil && runtime.GOOS != "windows" {
		return exit.Internalf("cannot protect the private authentication directory: %s", err)
	}
	data, err := json.Marshal(stored)
	if err != nil {
		return exit.Internalf("cannot encode the machine credential: %s", err)
	}
	temporary, err := os.CreateTemp(dir, ".machine-*.tmp")
	if err != nil {
		return exit.Internalf("cannot stage the machine credential: %s", err)
	}
	temporaryPath := temporary.Name()
	defer os.Remove(temporaryPath)
	if err := temporary.Chmod(0o600); err == nil {
		_, err = temporary.Write(append(data, '\n'))
	}
	if err == nil {
		err = temporary.Sync()
	}
	if closeErr := temporary.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return exit.Internalf("cannot write the machine credential: %s", err)
	}
	if err := os.Rename(temporaryPath, m.path); err != nil {
		return exit.Internalf("cannot commit the machine credential: %s", err)
	}
	return nil
}

func (m *Manager) post(ctx context.Context, path string, body, out any) *exit.Error {
	if problem := validateHub(m.hub); problem != nil {
		return problem
	}
	encoded, err := json.Marshal(body)
	if err != nil {
		return exit.Internalf("cannot encode the authentication request: %s", err)
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, m.hub+path, bytes.NewReader(encoded))
	if err != nil {
		return exit.Usagef("%q is not a usable Tensorhub URL: %s", m.hub, err)
	}
	request.Header.Set("Accept", "application/json")
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("User-Agent", "cozy-auth/1")
	response, err := m.http.Do(request)
	if err != nil {
		return hub.TransportFailure(m.hub, err)
	}
	defer response.Body.Close()
	data, err := io.ReadAll(io.LimitReader(response.Body, maxAnswer+1))
	if err != nil || len(data) > maxAnswer {
		return exit.Named(exit.Internal, "auth.unreadable_answer", "Tensorhub returned an unreadable authentication answer")
	}
	if response.StatusCode >= 300 {
		problem := authRefusal(response.StatusCode, data)
		if problem.Code == exit.Credential {
			problem.WithNext(m.loginCommand())
		}
		return problem
	}
	if err := json.Unmarshal(data, out); err != nil {
		return exit.Named(exit.Internal, "auth.unreadable_answer", "Tensorhub returned an invalid authentication answer")
	}
	return nil
}

func validateHub(raw string) *exit.Error {
	parsed, err := url.Parse(raw)
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" ||
		parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return exit.Usagef("%q is not an HTTP Tensorhub origin", raw)
	}
	return nil
}

func authRefusal(status int, data []byte) *exit.Error {
	var answer struct {
		Error struct {
			Code     string `json:"code"`
			Message  string `json:"message"`
			Remedy   string `json:"remedy"`
			Metadata struct {
				RetryAfterSeconds int `json:"retry_after_seconds"`
			} `json:"metadata"`
		} `json:"error"`
	}
	if json.Unmarshal(data, &answer) != nil || answer.Error.Code == "" {
		answer.Error.Code = "auth.refused"
		answer.Error.Message = "Tensorhub refused authentication"
	}
	code := exit.Validation
	switch {
	case status == http.StatusUnauthorized || status == http.StatusForbidden:
		code = exit.Credential
	case status == http.StatusTooManyRequests || status >= 500:
		code = exit.Unavailable
	}
	problem := exit.Named(code, answer.Error.Code, "%s", answer.Error.Message)
	if answer.Error.Metadata.RetryAfterSeconds > 0 {
		problem.WithRemedy("retry after %d seconds", answer.Error.Metadata.RetryAfterSeconds)
	} else if answer.Error.Remedy != "" {
		problem.WithRemedy("%s", answer.Error.Remedy)
	}
	return problem
}

// keyRefused names a device-key sign-in the Hub refuses for what it is: this computer's key
// was revoked or is unknown there, not a wrong password.
func (m *Manager) keyRefused(problem *exit.Error) *exit.Error {
	if problem.Code != exit.Credential {
		return problem
	}
	return exit.Named(exit.Credential, "auth.device_key_refused",
		"%s no longer accepts this computer's sign-in key (revoked or unknown there)", m.hub).WithNext(m.loginCommand())
}

// loginCommand names the login that creates this origin's credential. The default
// hub needs no selection; any other is named, since the credential is per origin.
func (m *Manager) loginCommand() string {
	if current := config.Frozen(); current.HubURLSource != "flag" && (m.hub == current.HubURL || current.HubURL == "") {
		return "cozy auth login <email>"
	}
	return "cozy auth login <email> --tensorhub=" + config.Frozen().HubLabel(m.hub)
}

// Invalidate drops the cached session so the next call mints a fresh bearer from
// the durable machine key. The hub client calls it when the hub refuses the
// current bearer before its expiry -- a restarted hub forgets our session
// without expiring our copy of it.
func (m *Manager) Invalidate() {
	m.mu.Lock()
	_ = os.Remove(m.sessionPath())
	m.session = Session{}
	m.mu.Unlock()
}
