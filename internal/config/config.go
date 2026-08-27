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
//
// LAYERED SOURCES (cl-029). A local product's configuration lives in a FILE, not a shell
// profile. Load composes, in precedence order (later wins):
//
//	defaults -> $COZY_HOME/config.yaml -> .env (cwd) -> process env -> CLI flags
//
// The file admits a CLOSED key set of flat `key: value` scalars — an unknown key refuses
// naming the known set, exactly as an unknown document field does. The .env and process
// layers admit only the same env names this reader has always had; the CLI layer is the
// manifest's own flags, applied by their verbs after Load. Every value keeps its source
// (`default | file | dotenv | env`) so an operator debugging a 401 knows which layer
// spoke last.
package config

import (
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/secret"
)

// Inherited is the CLOSED set of variables a child process may see from this process's
// own environment. Everything else a child needs is IMPOSED by its launcher as an
// explicit value (COZY_HOME, the device grant, the socket) — never inherited.
// Class-A base inherit list — tracker-v2/spawn-allowlists.md (#616.d) is the authority;
// keep equal to that row (Tool() adds NO_COLOR=1 per the same row).
var Inherited = []string{"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SSL_CERT_FILE", "SSL_CERT_DIR"}

// DefaultHubURL is where the catalog verbs look when nothing says otherwise. There is
// no deployed tensorhub yet (Launch 1 is being built), so the default names a hub on
// this host rather than inventing a production hostname a user cannot reach. `cozy hub
// status` always prints the URL AND where it came from, so the default is never silent.
const DefaultHubURL = "http://127.0.0.1:8080"

// DefaultPort is the loopback bind `cozy up` uses (cozy-creator.md: 2699, loopback
// only). It lives here — with the config authority that applies it — after cl-028
// found its previous spelling was a constant nobody read beside three literal 2699s.
const DefaultPort = 2699

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

	// LocalRateMicroUSDPerHour is the rate a user CONFIGURED for their own machine
	// (COZY_LOCAL_RATE_MICRO_USD_PER_HOUR), as an integer of micro-USD because a cost
	// fact that cannot be canonicalized cannot be journaled (cr-009's Budget rule).
	//
	// ZERO MEANS ABSENT, and absent means the bill is not rendered at all. A local job
	// costs electricity nobody metered, so `$0.00` would be a fabricated fact — the one
	// thing cl-004 says a rateless job must never print.
	LocalRateMicroUSDPerHour int64
	LocalRateSource          string

	// Bootstrap is the per-spawn worker bootstrap credential a LAUNCHER imposed on this
	// process (COZY_BOOTSTRAP_CREDENTIAL, #449) — present only in a spawned worker,
	// never inherited onward.
	Bootstrap secret.Value

	inherited []string // the allowlisted snapshot, captured at Load
}

var frozen Config

// FileName is the local product's configuration file, under $COZY_HOME.
const FileName = "config.yaml"

// fileKeys is the CLOSED key set config.yaml admits, each with the shape a refusal
// renders. `local_rate_micro_usd_per_hour` lives here FIRST-CLASS (cl-029): it is a user
// VALUE, not a credential, and a file is its primary home. The env spelling stays
// admitted above it — an implementer's recorded call, because the harness and existing
// dev roots already speak it — and `cozy job status` renders whichever source won.
var fileKeys = map[string]string{
	"tensorhub_url":                 "the catalog host the hub verbs talk to",
	"tensorhub_token":               "the Launch-1 static admin token (writes only)",
	"tfs":                           "the tensorfs CLI this binary delegates the byte plane to",
	"local_rate_micro_usd_per_hour": "your machine's own rate, integer micro-USD (250000 = $0.25/hour)",
	"port":                          "the loopback client API port",
	"yield":                         "the GPU yield policy: smart | always | never",
}

func knownFileKeys() string {
	keys := make([]string, 0, len(fileKeys))
	for k := range fileKeys {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return strings.Join(keys, ", ")
}

// dotenvNames is what the `.env` layer admits: exactly the env names this reader speaks,
// minus the per-spawn bootstrap credential — that one is IMPOSED by a launcher on the
// child it just created and has no business in a file another process could share.
// Unknown keys in .env are IGNORED, not refused: the file is a directory-level
// convention other tools share, and this reader takes only its own names from it.
var dotenvNames = []string{
	"COZY_HOME", "COZY_TFS", "COZY_LOCAL_RATE_MICRO_USD_PER_HOUR",
	"TENSORHUB_URL", "TENSORHUB_TOKEN",
}

// Load composes the layered sources ONCE and freezes them. Calling it twice returns the
// same value: a second read cannot see a different world than the first.
func Load() (Config, *exit.Error) {
	if frozen.Home != "" {
		return frozen, nil
	}
	c := Config{Port: DefaultPort, Yield: "smart",
		HubURL: DefaultHubURL, HubURLSource: "default", HubTokenSource: "unset",
		Tfs: "tfs", TfsSource: "default"}
	c.LocalRateSource = "unset"

	dotenv, e := readDotEnv(".env")
	if e != nil {
		return Config{}, e
	}
	// layered answers one name: process env beats .env; "" means neither layer spoke.
	layered := func(name string) (string, string) {
		if v := strings.TrimSpace(os.Getenv(name)); v != "" {
			return v, "env"
		}
		if v := strings.TrimSpace(dotenv[name]); v != "" {
			return v, "dotenv"
		}
		return "", ""
	}

	// HOME FIRST, because the file layer lives under it. It has no config.yaml spelling
	// for the same reason.
	if v, _ := layered("COZY_HOME"); v != "" {
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

	// THE FILE LAYER: above defaults, below everything that can vary per invocation.
	file, e := readConfigFile(filepath.Join(c.Home, FileName))
	if e != nil {
		return Config{}, e
	}
	if v := file["tensorhub_url"]; v != "" {
		c.HubURL, c.HubURLSource = strings.TrimRight(v, "/"), "file"
	}
	if v := file["tensorhub_token"]; v != "" {
		c.HubToken, c.HubTokenSource = secret.New(v), "file"
	}
	if v := file["tfs"]; v != "" {
		c.Tfs, c.TfsSource = v, "file"
	}
	if v := file["local_rate_micro_usd_per_hour"]; v != "" {
		n, e := parseRate(v, FileName+" local_rate_micro_usd_per_hour")
		if e != nil {
			return Config{}, e
		}
		c.LocalRateMicroUSDPerHour, c.LocalRateSource = n, "file"
	}
	if v := file["port"]; v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n <= 0 || n > 65535 {
			return Config{}, exit.Usagef("%s port %q is not a TCP port", FileName, v)
		}
		c.Port = n
	}
	if v := file["yield"]; v != "" {
		switch v {
		case "smart", "always", "never":
			c.Yield = v
		default:
			return Config{}, exit.Usagef("%s yield %q is not smart|always|never", FileName, v)
		}
	}

	// THE ENV LAYERS, over the file: .env in this directory, then the process env.
	if v, src := layered("COZY_TFS"); v != "" {
		c.Tfs, c.TfsSource = v, src
	}
	if v, src := layered("COZY_LOCAL_RATE_MICRO_USD_PER_HOUR"); v != "" {
		n, e := parseRate(v, "COZY_LOCAL_RATE_MICRO_USD_PER_HOUR="+v)
		if e != nil {
			return Config{}, e
		}
		c.LocalRateMicroUSDPerHour, c.LocalRateSource = n, src
	}
	if v, src := layered("TENSORHUB_URL"); v != "" {
		c.HubURL, c.HubURLSource = strings.TrimRight(v, "/"), src
	}
	if v, src := layered("TENSORHUB_TOKEN"); v != "" {
		c.HubToken, c.HubTokenSource = secret.New(v), src
	}
	// The PER-SPAWN worker bootstrap credential (#449): imposed by the launcher on the
	// child it just created, read here because this file is the one env reader, consumed
	// by the worker side of the protocol. Deliberately NOT in the inherited allowlist and
	// NOT a file or .env key — it exists for exactly one process.
	if v := os.Getenv("COZY_BOOTSTRAP_CREDENTIAL"); strings.TrimSpace(v) != "" {
		c.Bootstrap = secret.New(v)
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

func parseRate(v, spelled string) (int64, *exit.Error) {
	n, err := strconv.ParseInt(v, 10, 64)
	if err != nil || n < 0 {
		return 0, exit.Usagef("%s is not a non-negative integer of micro-USD", spelled).
			WithRemedy("these documents are integer-only; 250000 is $0.25/hour")
	}
	return n, nil
}

// readConfigFile reads $COZY_HOME/config.yaml: flat `key: value` scalars, `#` comments,
// blank lines, and NOTHING else. The key set is CLOSED — an unknown key refuses naming
// the known set — and the parser is deliberately this product's own: a general YAML
// loader would admit structure this vocabulary has no meaning for, and YAML parsing
// belongs to internal/video (the fence says so).
func readConfigFile(path string) (map[string]string, *exit.Error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, exit.Internalf("cannot read %s: %s", path, err)
	}
	out := map[string]string{}
	for i, line := range strings.Split(string(data), "\n") {
		s := strings.TrimSpace(line)
		if s == "" || strings.HasPrefix(s, "#") {
			continue
		}
		key, value, ok := strings.Cut(s, ":")
		key = strings.TrimSpace(key)
		if !ok || key == "" || strings.ContainsAny(key, " \t") {
			return nil, exit.Usagef("%s line %d is not a flat `key: value` scalar: %q",
				path, i+1, s).
				WithRemedy("this file admits: %s", knownFileKeys())
		}
		if _, known := fileKeys[key]; !known {
			return nil, exit.Usagef("%s names %q, which this product does not read", path, key).
				WithRemedy("it reads: %s", knownFileKeys())
		}
		if _, dup := out[key]; dup {
			return nil, exit.Usagef("%s names %q twice; a value has one spelling", path, key)
		}
		out[key] = trimQuotes(strings.TrimSpace(value))
	}
	return out, nil
}

// readDotEnv reads `.env` in the working directory: `KEY=VALUE` lines (an optional
// `export ` survives), `#` comments and blank lines. Only this reader's own names are
// taken; everything else in the file belongs to other tools and is ignored.
func readDotEnv(path string) (map[string]string, *exit.Error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		// Unreadable is not absent: a .env that exists and cannot be read would make
		// this layer silently vanish, which is a debugging session nobody wants.
		return nil, exit.Internalf("cannot read %s: %s", path, err)
	}
	admitted := map[string]bool{}
	for _, name := range dotenvNames {
		admitted[name] = true
	}
	out := map[string]string{}
	for _, line := range strings.Split(string(data), "\n") {
		s := strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(line), "export "))
		if s == "" || strings.HasPrefix(s, "#") {
			continue
		}
		key, value, ok := strings.Cut(s, "=")
		key = strings.TrimSpace(key)
		if !ok || !admitted[key] {
			continue
		}
		out[key] = trimQuotes(strings.TrimSpace(value))
	}
	return out, nil
}

func trimQuotes(v string) string {
	if len(v) >= 2 && (v[0] == '"' && v[len(v)-1] == '"' || v[0] == '\'' && v[len(v)-1] == '\'') {
		return v[1 : len(v)-1]
	}
	return v
}

// Child builds a child process's whole environment: the allowlisted snapshot plus the
// exact values the launcher imposes. The result is sorted and deduplicated on the
// variable name, imposed values winning — a child never sees two spellings of one name.
func (c Config) Child(imposed ...string) []string {
	// THE BYTE-PLANE DOOR travels with the launcher, because it is a resolved TOOL PATH
	// this process already decided and not a decision a child may make again. Without it a
	// detached `cozy up` — or a driver-started one — came up with no `tfs` on its PATH and
	// silently could not root a job's declared checkpoints. It is imposed only when it was
	// configured; the default (`tfs`, found on PATH) needs no help.
	if c.TfsSource == "env" && c.Tfs != "" {
		imposed = append([]string{"COZY_TFS=" + c.Tfs}, imposed...)
	}
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
