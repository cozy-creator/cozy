package api

import (
	"fmt"
	"net/http"
	"os"
	"runtime"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/secret"
)

// Credentials is the per-launch CLI credential. It rides the daemon's own lifetime
// ownership record (`daemon.lock`, mode 0600) — never argv or inherited environment —
// and rotates with every launch because daemon.Hold rewrites that record (cl-116).
type Credentials struct {
	CLI secret.Value
}

// Mint generates a fresh credential and appends it to the daemon record the caller
// already holds. Hold truncated the body this launch, so exactly one token line exists.
func Mint(l home.Layout) (Credentials, *exit.Error) {
	c := Credentials{CLI: secret.Mint()}
	f, err := os.OpenFile(l.Daemon, os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return Credentials{}, exit.Internalf("cannot write the local client credential: %s", err)
	}
	defer f.Close()
	// Chmod anyway: a record created by an older build was 0644, and a credential
	// readable by the group would be exactly the leak this file prevents.
	if err := f.Chmod(0o600); err != nil {
		return Credentials{}, exit.Internalf("cannot restrict the daemon record: %s", err)
	}
	if _, err := fmt.Fprintf(f, "token=%s\n", c.CLI.Reveal()); err != nil {
		return Credentials{}, exit.Internalf("cannot write the local client credential: %s", err)
	}
	return c, nil
}

// Admits checks the one local bearer in constant time.
func (c Credentials) Admits(presented string) bool {
	return c.CLI.Equal(presented)
}

// AdmitsCLI is the narrower authority check for operations that may read the caller's
// filesystem. A browser bearer can drive ordinary request lifecycle routes, but it must
// never name host paths; a future browser asset surface has to upload selected bytes.
func (c Credentials) AdmitsCLI(presented string) bool { return c.CLI.Equal(presented) }

// ClientCredential is the CLI side of the handoff: the token line of the daemon's own
// 0600 record (cl-116; cl-010's separate client.cred file is deleted). It is
// deliberately here and not in the client package: this file is the credential's
// carrier site, so the raw value is read where it becomes a carrier and nowhere else —
// the `secret` fence family holds that.
//
// A record without a token means the Calcifer is not running or has not finished
// launching on this root. That is exactly the exit-9 condition every server-backed verb
// already shares, so it refuses with the same remedy.
func ClientCredential(l home.Layout) (secret.Value, *exit.Error) {
	info, err := os.Stat(l.Daemon)
	if err != nil {
		return secret.Value{}, exit.Unavailablef(
			"no daemon record at %s", l.Daemon).
			WithRemedy("the daemon writes it at launch; retry the command to auto-start or reconnect").
			WithNext("cozy run list")
	}
	// The mode is CHECKED, not assumed. A record that became group- or world-readable
	// (an inherited umask, a careless copy) is a refusal: reading it anyway would be
	// the client agreeing to a leak the server tried to prevent. On Windows, Unix mode
	// bits are a fiction (Go reports 0666 for every file) — the boundary there is the
	// user profile's ACL, which already scopes COZY_HOME to the user.
	if perm := info.Mode().Perm(); perm&0o077 != 0 && runtime.GOOS != "windows" {
		return secret.Value{}, exit.New(exit.Credential,
			"%s is mode %#o; the daemon record carrying the client credential is 0600 or it is not used", l.Daemon, perm).
			WithRemedy("run `cozy down`, fix the file mode, then retry; every daemon launch mints a fresh credential")
	}
	data, err := os.ReadFile(l.Daemon)
	if err != nil {
		return secret.Value{}, exit.New(exit.Credential,
			"the local client credential is unreadable: %s", err)
	}
	for _, line := range strings.Split(string(data), "\n") {
		if raw, ok := strings.CutPrefix(strings.TrimSpace(line), "token="); ok {
			if v := secret.New(raw); v.Present() {
				return v, nil
			}
		}
	}
	return secret.Value{}, exit.Unavailablef(
		"the daemon record at %s carries no client credential yet", l.Daemon).
		WithRemedy("the daemon appends it once its API is up; retry the command").
		WithNext("cozy run list")
}

// Authorize puts a credential onto one outbound request. It is the ONLY place a local
// client credential becomes a header, which is why it lives beside the writer.
func Authorize(r *http.Request, v secret.Value) {
	r.Header.Set("Authorization", "Bearer "+v.Reveal())
}
