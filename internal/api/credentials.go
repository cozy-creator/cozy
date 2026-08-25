package api

import (
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/home"
	"github.com/cozy-creator/cozy-creator-v2/internal/secret"
)

// Two credentials, minted per launch, and they are DIFFERENT on purpose (cozy-creator.md:
// "CLI clients use a permission-restricted local credential, never a browser credential").
//
//   - Browser: handed to the stub page through the `--open` URL's FRAGMENT. A fragment is
//     never sent to a server, never lands in an access log, and never leaks through
//     `Referer` — which is the whole reason it is not a query parameter. The page scrubs it
//     with history.replaceState so it does not survive in the address bar or in history.
//   - CLI: handed over through a 0600 file in the local root. An OS-protected handoff,
//     because argv is world-readable on this planet (cl-011's rule, `secret` fence family)
//     and an environment variable is inherited by every child the CLI spawns.
//
// Both die with the process that minted them. There is no persistence, no refresh, and no
// revocation list — a restarted LocalService has new credentials and every old one is
// worthless, which is the cheapest correct revocation there is.
//
// What is deliberately NOT here: the session-cookie choreography (fragment secret → one-time
// exchange → HttpOnly; SameSite=Strict cookie). Its consumer is cl-007's real UI and it
// lands with it. Everything around it — loopback, Host allowlist, Origin checks, opaque
// media, no token in argv or logs — is launch surface NOW.

// Credentials is the per-launch credential pair.
type Credentials struct {
	Browser secret.Value
	CLI     secret.Value
}

// Mint generates a fresh pair and writes the CLI half to its 0600 file. The browser half
// is written nowhere: it exists in this process and in the URL the user's browser was
// handed, and in no third place.
func Mint(l home.Layout) (Credentials, *exit.Error) {
	c := Credentials{Browser: secret.Mint(), CLI: secret.Mint()}
	// O_EXCL is not used: a stale file from a dead service is a leftover, and the
	// service lock already proved no live owner exists on this root. 0600 is the point.
	f, err := os.OpenFile(l.Client, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o600)
	if err != nil {
		return Credentials{}, exit.Internalf("cannot write the local client credential: %s", err)
	}
	defer f.Close()
	// Chmod anyway: O_CREAT's mode is masked by umask, and a credential readable by the
	// group because of an inherited umask would be exactly the bug this file prevents.
	if err := f.Chmod(0o600); err != nil {
		return Credentials{}, exit.Internalf("cannot restrict the client credential file: %s", err)
	}
	if _, err := fmt.Fprintln(f, c.CLI.Reveal()); err != nil {
		return Credentials{}, exit.Internalf("cannot write the local client credential: %s", err)
	}
	return c, nil
}

// Admits answers whether a presented bearer is either credential. Constant time on both
// comparisons, and the raw values are never one side of them.
func (c Credentials) Admits(presented string) bool {
	// Both are evaluated: short-circuiting on the first match would leak, by timing,
	// WHICH credential was presented.
	browser := c.Browser.Equal(presented)
	cli := c.CLI.Equal(presented)
	return browser || cli
}

// ClientCredential is the CLI side of the 0600 handoff (cl-010, the file's first reader).
// It is deliberately here and not in the client package: this file is the credential's
// carrier site, so the raw value is read where it becomes a carrier and nowhere else —
// the `secret` fence family holds that.
//
// A missing file means the LocalService is not running or is running on another root.
// That is exactly the exit-9 condition every server-backed verb already shares, so it
// refuses with the same remedy rather than inventing a credential vocabulary.
func ClientCredential(l home.Layout) (secret.Value, *exit.Error) {
	info, err := os.Stat(l.Client)
	if err != nil {
		return secret.Value{}, exit.Unavailablef(
			"no local client credential at %s", l.Client).
			WithRemedy("the LocalService writes it at launch and it dies with the process").
			WithNext("cozy up", "cozy status")
	}
	// The mode is CHECKED, not assumed. A credential that became group- or
	// world-readable (an inherited umask, a careless copy) is a refusal: reading it
	// anyway would be the client agreeing to a leak the server tried to prevent.
	if perm := info.Mode().Perm(); perm&0o077 != 0 {
		return secret.Value{}, exit.New(exit.Credential,
			"%s is mode %#o; the local client credential is 0600 or it is not used", l.Client, perm).
			WithRemedy("restart the LocalService: every launch mints a fresh pair").
			WithNext("cozy down", "cozy up")
	}
	data, err := os.ReadFile(l.Client)
	if err != nil {
		return secret.Value{}, exit.New(exit.Credential,
			"the local client credential is unreadable: %s", err)
	}
	v := secret.New(string(data))
	if !v.Present() {
		return secret.Value{}, exit.New(exit.Credential,
			"the local client credential at %s is empty", l.Client).
			WithNext("cozy down", "cozy up")
	}
	return v, nil
}

// Authorize puts a credential onto one outbound request. It is the ONLY place a local
// client credential becomes a header, which is why it lives beside the writer.
func Authorize(r *http.Request, v secret.Value) {
	r.Header.Set("Authorization", "Bearer "+v.Reveal())
}

// OpenURL is what `--open` hands the browser: the stub page with the browser credential
// in the FRAGMENT. The query string is deliberately empty — a token there would reach the
// server, the access log, and every `Referer` the page emits.
func (c Credentials) OpenURL(addr string) string {
	_, port, err := net.SplitHostPort(addr)
	if err != nil {
		port = "2699"
	}
	return "http://127.0.0.1:" + port + "/#t=" + url.QueryEscape(c.Browser.Reveal())
}
