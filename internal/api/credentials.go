package api

import (
	"fmt"
	"net/http"
	"os"
	"runtime"

	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/home"
	"github.com/cozy-creator/cozy-creator/internal/secret"
)

// Credentials is the per-launch CLI credential. It is handed over through a
// 0600 file, never argv or inherited environment, and dies with the controller.
type Credentials struct {
	CLI secret.Value
}

// Mint generates a fresh credential and writes it to its 0600 handoff file.
func Mint(l home.Layout) (Credentials, *exit.Error) {
	c := Credentials{CLI: secret.Mint()}
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

// Admits checks the one local bearer in constant time.
func (c Credentials) Admits(presented string) bool {
	return c.CLI.Equal(presented)
}

// AdmitsCLI is the narrower authority check for operations that may read the caller's
// filesystem. A browser bearer can drive ordinary request lifecycle routes, but it must
// never name host paths; a future browser asset surface has to upload selected bytes.
func (c Credentials) AdmitsCLI(presented string) bool { return c.CLI.Equal(presented) }

// ClientCredential is the CLI side of the 0600 handoff (cl-010, the file's first reader).
// It is deliberately here and not in the client package: this file is the credential's
// carrier site, so the raw value is read where it becomes a carrier and nowhere else —
// the `secret` fence family holds that.
//
// A missing file means the local controller is not running or is running on another root.
// That is exactly the exit-9 condition every server-backed verb already shares, so it
// refuses with the same remedy rather than inventing a credential vocabulary.
func ClientCredential(l home.Layout) (secret.Value, *exit.Error) {
	info, err := os.Stat(l.Client)
	if err != nil {
		return secret.Value{}, exit.Unavailablef(
			"no local client credential at %s", l.Client).
			WithRemedy("the controller writes it at launch; retry the command to auto-start or reconnect").
			WithNext("cozy invoke list")
	}
	// The mode is CHECKED, not assumed. A credential that became group- or
	// world-readable (an inherited umask, a careless copy) is a refusal: reading it
	// anyway would be the client agreeing to a leak the server tried to prevent.
	// On Windows, Unix mode bits are a fiction (Go reports 0666 for every file) — the
	// boundary there is the user profile's ACL, which already scopes COZY_HOME to the
	// user, so the bits are not consulted.
	if perm := info.Mode().Perm(); perm&0o077 != 0 && runtime.GOOS != "windows" {
		return secret.Value{}, exit.New(exit.Credential,
			"%s is mode %#o; the local client credential is 0600 or it is not used", l.Client, perm).
			WithRemedy("run `cozy down`, fix the file mode, then retry; every controller launch mints a fresh credential")
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
			WithNext("cozy down")
	}
	return v, nil
}

// Authorize puts a credential onto one outbound request. It is the ONLY place a local
// client credential becomes a header, which is why it lives beside the writer.
func Authorize(r *http.Request, v secret.Value) {
	r.Header.Set("Authorization", "Bearer "+v.Reveal())
}
