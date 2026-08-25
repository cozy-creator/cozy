// Package config is THE ONE environment reader in this binary (cl-001). The process
// entrypoint calls Load exactly once; every later read is of a FROZEN typed value, so a
// mutation after startup is structurally invisible and no package can disagree with
// another about what the environment said.
//
// The env-read fence (scripts/fence.py, family `env`) is what keeps that true: os.Getenv,
// os.LookupEnv, os.Environ and syscall.Getenv appear in this file and nowhere else.
//
// It is also the child-environment allowlist. cozy-creator starts endpoint processes; a
// child inherits exactly what Child() names and nothing else (cozy-creator.md: "child env
// allowlisted before the runtime starts"), which is why os.Environ() has no second caller.
package config

import (
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/secret"
)

// Inherited is the CLOSED set of variables a child process may see from this process's
// own environment. Everything else a child needs is IMPOSED by its launcher as an
// explicit value (COZY_HOME, the device grant, the socket) — never inherited.
var Inherited = []string{"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SSL_CERT_FILE", "SSL_CERT_DIR"}

// DefaultHubURL is where the catalog verbs look when nothing says otherwise. There is
// no deployed tensorhub yet (Launch 1 is being built), so the default names a hub on
// this host rather than inventing a production hostname a user cannot reach. `cozy hub
// status` always prints the URL AND where it came from, so the default is never silent.
const DefaultHubURL = "http://127.0.0.1:8080"

// Config is the frozen typed value. Every field is decided once, at Load.
type Config struct {
	Home string // the local root: $COZY_HOME, or ~/.cozy
	// Port is the local client API bind port (loopback only). cl-006 serves routes on it;
	// cl-001 binds it so `cozy status` can tell up from down without a sidecar file.
	Port int
	// Yield is the GPU yield policy: smart | always | never (cozy-creator.md).
	Yield string

	// HubURL is the catalog host the cl-011 verbs read and write (TENSORHUB_URL).
	HubURL string
	// HubToken is the ONE static admin token Launch-1 tensorhub admits writes with
	// (TENSORHUB_TOKEN). Reads need no credential. There is no login act and no
	// identity plane until th-031 — the env carries a VALUE, never a decision.
	HubToken secret.Value
	// Source of each, for rendering: "default" or "env". An operator debugging a 401
	// needs to know which file lied to them, and a value with no provenance is a guess.
	HubURLSource   string
	HubTokenSource string

	// Tfs is the tensorfs CLI this binary asks every byte-plane question of (COZY_TFS).
	// cozy-creator owns no byte plane (boundaries.md); it coordinates one.
	Tfs       string
	TfsSource string

	inherited []string // the allowlisted snapshot, captured at Load
}

var frozen Config

// Load reads the environment ONCE and freezes it. Calling it twice returns the same
// value: a second read cannot see a different world than the first.
func Load() (Config, *exit.Error) {
	if frozen.Home != "" {
		return frozen, nil
	}
	c := Config{Port: 2699, Yield: "smart",
		HubURL: DefaultHubURL, HubURLSource: "default", HubTokenSource: "unset",
		Tfs: "tfs", TfsSource: "default"}

	if v := strings.TrimSpace(os.Getenv("COZY_TFS")); v != "" {
		c.Tfs, c.TfsSource = v, "env"
	}

	if v := strings.TrimSpace(os.Getenv("TENSORHUB_URL")); v != "" {
		c.HubURL, c.HubURLSource = strings.TrimRight(v, "/"), "env"
	}
	if v := os.Getenv("TENSORHUB_TOKEN"); strings.TrimSpace(v) != "" {
		c.HubToken, c.HubTokenSource = secret.New(v), "env"
	}

	if v := strings.TrimSpace(os.Getenv("COZY_HOME")); v != "" {
		abs, err := filepath.Abs(v)
		if err != nil {
			return Config{}, exit.Internalf("COZY_HOME %q is not resolvable: %s", v, err)
		}
		c.Home = abs
	} else {
		h, err := os.UserHomeDir()
		if err != nil {
			return Config{}, exit.Internalf("no home directory and COZY_HOME is unset: %s", err)
		}
		c.Home = filepath.Join(h, ".cozy")
	}

	for _, name := range Inherited {
		if v, ok := os.LookupEnv(name); ok {
			c.inherited = append(c.inherited, name+"="+v)
		}
	}
	sort.Strings(c.inherited)
	frozen = c
	return c, nil
}

// Child builds a child process's whole environment: the allowlisted snapshot plus the
// exact values the launcher imposes. The result is sorted and deduplicated on the
// variable name, imposed values winning — a child never sees two spellings of one name.
func (c Config) Child(imposed ...string) []string {
	seen := map[string]string{}
	for _, kv := range c.inherited {
		name, value, _ := strings.Cut(kv, "=")
		seen[name] = value
	}
	for _, kv := range imposed {
		name, value, _ := strings.Cut(kv, "=")
		seen[name] = value
	}
	out := make([]string, 0, len(seen))
	for name, value := range seen {
		out = append(out, name+"="+value)
	}
	sort.Strings(out)
	return out
}

// Tool is the environment for a short-lived tool this process shells out to (uv,
// nvidia-smi, an endpoint's describe). Same allowlist, plus NO_COLOR so a tool's output
// is parseable.
func (c Config) Tool(imposed ...string) []string {
	return c.Child(append([]string{"NO_COLOR=1"}, imposed...)...)
}

// Frozen is the value Load produced. It is how a package deep in a call chain reads
// configuration without reading the environment; if Load has not run, it is the zero
// value and every path that needs a root fails loudly rather than inventing one.
func Frozen() Config { return frozen }
