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
	if problem := m.post(ctx, "/v1/auth/device-keys/enroll/begin", map[string]string{
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
	if problem := m.post(ctx, "/v1/auth/device-keys/enroll/finish", map[string]string{
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

// DeleteCredential erases exactly the credential whose server revocation was
// confirmed. A mismatched concurrent login is retained.
func (m *Manager) DeleteCredential(deviceKeyID string) *exit.Error {
	m.mu.Lock()
	defer m.mu.Unlock()
	stored, _, problem := m.load()
	if problem != nil {
		return problem
	}
	if stored.DeviceKeyID != deviceKeyID {
		return exit.Named(exit.Internal, "auth.machine_changed",
			"the local machine credential changed while logout was in progress")
	}
	if err := os.Remove(m.path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return exit.Internalf("cannot erase the machine credential: %s", err)
	}
	m.session = Session{}
	return nil
}

// Manager owns one Tensorhub origin's machine key and memory-only access token.
// Constructing one performs no I/O or network work.
type Manager struct {
	hub     string
	path    string
	http    *http.Client
	mu      sync.Mutex
	session Session
	now     func() time.Time
}

func New(cfg config.Config) *Manager {
	origin := strings.TrimRight(strings.TrimSpace(cfg.HubURL), "/")
	sum := sha256.Sum256([]byte(origin))
	return &Manager{
		hub:  origin,
		path: filepath.Join(cfg.Home, "auth", hex.EncodeToString(sum[:])+".json"),
		http: &http.Client{Timeout: hub.Timeout},
		now:  time.Now,
	}
}

// AccessToken satisfies hub.TokenSource. A daemon reuses the cached token while it is
// comfortably live; a short CLI process performs one cheap login exchange.
func (m *Manager) AccessToken(ctx context.Context) (secret.Value, *exit.Error) {
	session, problem := m.Authenticate(ctx)
	return session.AccessToken, problem
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
	if m.session.AccessToken.Present() && m.session.ExpiresAt.After(m.now().Add(30*time.Second)) {
		return m.session, nil
	}
	stored, private, problem := m.load()
	if problem != nil {
		return Session{}, problem
	}
	var begun struct {
		ChallengeID string `json:"challenge_id"`
		Challenge   string `json:"challenge"`
		ExpiresAt   string `json:"expires_at"`
	}
	if problem := m.post(ctx, "/v1/auth/device-keys/login/begin", map[string]string{
		"device_key_id": stored.DeviceKeyID,
	}, &begun); problem != nil {
		return Session{}, problem
	}
	challenge, problem := challengeBytes(begun.Challenge)
	if problem != nil || begun.ChallengeID == "" {
		return Session{}, exit.Named(exit.Internal, "auth.unreadable_challenge",
			"Tensorhub returned an invalid machine-login challenge")
	}
	signature := sign(private, loginDomain, challenge)
	var answer tokenAnswer
	started := m.now()
	if problem := m.post(ctx, "/v1/auth/device-keys/login/finish", map[string]string{
		"challenge_id": begun.ChallengeID,
		"signature":    rawBase64.EncodeToString(signature),
	}, &answer); problem != nil {
		return Session{}, problem
	}
	m.session, problem = answer.session(stored.Email, started)
	return m.session, problem
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
	if problem := m.post(ctx, "/v1/auth/device-keys/enroll/begin", map[string]string{
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
	if problem := m.post(ctx, "/v1/auth/device-keys/enroll/finish", map[string]string{
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
			WithNext("cozy auth login <email>")
	}
	if err != nil {
		return credential{}, nil, exit.Named(exit.Credential, "auth.machine_key_unreadable",
			"the machine credential cannot be read: %s", err).
			WithNext("cozy auth login <email>")
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
			"the stored machine credential is invalid").WithNext("cozy auth login <email>")
	}
	seed, err := rawBase64.DecodeString(stored.PrivateKey)
	if stored.Version != credentialVersion || stored.Hub != m.hub || stored.DeviceKeyID == "" ||
		err != nil || len(seed) != ed25519.SeedSize {
		return credential{}, nil, exit.Named(exit.Credential, "auth.machine_key_invalid",
			"the stored machine credential is invalid").WithNext("cozy auth login <email>")
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
		return authRefusal(response.StatusCode, data)
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
	if code == exit.Credential {
		problem.WithNext("cozy auth login <email>")
	}
	return problem
}

// Invalidate drops the cached session so the next call mints a fresh bearer from
// the durable machine key. The hub client calls it when the hub refuses the
// current bearer before its expiry -- a restarted hub forgets our session
// without expiring our copy of it.
func (m *Manager) Invalidate() {
	m.mu.Lock()
	m.session = Session{}
	m.mu.Unlock()
}
