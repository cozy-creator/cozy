// Package config owns Cozy's one process-configuration read.
//
// Load resolves one private Kong grammar with no argv. Sources are, in order:
// defaults, $COZY_HOME/config.yaml, and process environment. The result is frozen for the process lifetime.
// Secrets therefore use Kong's resolver pipeline without becoming command-line
// flags.
package config

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/alecthomas/kong"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/secret"
	"github.com/robfig/cron/v3"
	"go.yaml.in/yaml/v3"
)

// Inherited is the complete set of environment variables a package process
// may inherit. Launchers impose every other value explicitly.
var Inherited = []string{"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SSL_CERT_FILE", "SSL_CERT_DIR"}

const (
	// DefaultHubName names DefaultHubURL, the one Tensorhub most installations use.
	DefaultHubName = "tensorhub"
	DefaultHubURL  = "https://tensorhub.com"
	DefaultPort    = 8818
	FileName       = "config.yaml"
)

// Config is the frozen value consumed by the rest of the process.
type Config struct {
	Home string
	Port int
	// PortSource distinguishes the product default, which may fall back when
	// occupied, from an operator-selected port that must bind exactly.
	PortSource string
	Yield      string

	// HubURL is the Tensorhub origin this process's commands address: the current
	// hub (tensorhub_url, a name or URL) or the one-command --tensorhub selection.
	// Records keep their own origin; this is only the default for new work.
	HubURL       string
	HubName      string            // HubURL's name in Hubs, when it has one
	Hubs         map[string]string // named hubs (kubectl-style contexts), name -> origin
	HubURLSource string
	// ConfiguredHubURL is the current hub before any one-command --tensorhub: what
	// config.yaml or the environment selects.
	ConfiguredHubURL string
	// HubToken is HubURL's static operator bearer, if any. The configured token belongs
	// to ConfiguredHubURL and is never sent anywhere else (ForHub).
	HubToken              secret.Value
	HubTokenSource        string
	configuredToken       secret.Value
	configuredTokenSource string
	// tokenSettings names every place the operator token is set, so a refusal can say
	// exactly what to remove.
	tokenSettings []string

	HuggingFaceToken       secret.Value
	HuggingFaceTokenSource string
	CivitaiToken           secret.Value
	CivitaiTokenSource     string

	// MachineGPUBudget caps the GPU memory this computer's machine may use, verbatim from
	// `machine.gpu_budget`: its Runtime reads it as gpu.budget. Nil leaves the GPUs uncapped.
	MachineGPUBudget any

	Tfs       string
	TfsSource string
	// TensorFSRoot is the independent local TensorFS Store this Creator consumes
	// (proto-030/tfs-047): explicit configuration first, then TENSORFS_HOME, then
	// `~/.tensorfs`. It is never derived from COZY_HOME — TensorFS is not a
	// subdirectory of Creator.
	TensorFSRoot       string
	TensorFSRootSource string
	// TensorFSRegistry is an operator/test-only override. Ordinary imports use
	// the reviewed registry embedded by the installed TensorFS binary.
	TensorFSRegistry string

	LocalRateMicroUSDPerHour int64
	LocalRateSource          string
	// Development rentals use the Hub's registered debug image and this owner's
	// public SSH key. These defaults apply only when authoring a new acquisition.
	RentalsDevelopment  bool
	RentalsSSHPublicKey string

	// DaemonIdleShutdown is how long "nothing to manage" must stay true before the daemon
	// exits on its own. It is a debounce over an observed fact, never the decision; zero
	// keeps the daemon up until `cozy down`.
	DaemonIdleShutdown time.Duration

	// PlacementPrefer is the tier a rented run is placed under (placement-economics.md):
	// fast | balanced | cheap. Config only, never a flag.
	PlacementPrefer string
	// Digest is the sha256 of config.yaml's bytes, so a decision record can cite the
	// configuration it read.
	Digest string

	// MaintenanceGCCron is when the daemon runs the store's reclamation pass (owner ruling
	// 2026-09-02: repo-CAS garbage collection runs on a cron job). A cadence, never a
	// decision: what is reclaimed is TensorFS's call from its filesystem census. Standard
	// five-field cron; empty disables the scheduled pass (`cozy model remove` and `cozy
	// model gc` still reclaim on demand).
	MaintenanceGCCron string

	// PlayerURL is the page `cozy run play` links open: GitHub Pages' build of web/player
	// unless another build is configured.
	PlayerURL string

	inherited []string
}

// values is a config-only Kong grammar. It is never embedded in the public CLI
// grammar and its parser always receives nil argv.
type values struct {
	HubURL                   string `name:"tensorhub_url" default:"tensorhub"`
	HubToken                 string `name:"tensorhub_token"`
	HuggingFaceToken         string `name:"huggingface_token"`
	CivitaiToken             string `name:"civitai_token"`
	Tfs                      string `name:"tfs" default:"tfs"`
	TensorFSRoot             string `name:"tensorfs_root"`
	TensorFSRegistry         string `name:"tensorfs_registry"`
	LocalRateMicroUSDPerHour int64  `name:"local_rate_micro_usd_per_hour" default:"0"`
	RentalsDevelopment       bool   `name:"rentals_development" default:"true"`
	RentalsSSHPublicKey      string `name:"rentals_ssh_public_key"`
	DaemonIdleShutdownS      int64  `name:"daemon_idle_shutdown_s" default:"900"`
	MaintenanceGCCron        string `name:"maintenance_gc_cron" default:"0 3 * * *"`
	PlacementPrefer          string `name:"placement_prefer" default:"balanced"`
	PlayerURL                string `name:"player_url" default:"https://cozy-creator.github.io/cozy/play/"`
	Port                     int    `name:"port" default:"8818"`
	Yield                    string `name:"yield" default:"smart" enum:"smart,always,never"`
}

// Validate refuses only an empty location. Behaviour settings were admitted per key
// before resolution (admitBehaviour), so none of them can refuse a command.
func (v *values) Validate() error {
	if strings.TrimSpace(v.HubURL) == "" {
		return fmt.Errorf("tensorhub_url must not be empty")
	}
	if strings.TrimSpace(v.Tfs) == "" {
		return fmt.Errorf("tfs must not be empty")
	}
	return nil
}

// behaviour names the settings that only tune how Cozy behaves, each with the check its
// consumer needs. An unusable value is named once on stderr and the default applies;
// locations and credentials still refuse, because a guessed location acts somewhere else.
var behaviour = map[string]struct {
	spelled string
	admit   func(string) error
}{
	"local_rate_micro_usd_per_hour": {"local_rate_micro_usd_per_hour", nonNegative},
	"rentals_development":           {"rentals.development", boolean},
	"daemon_idle_shutdown_s":        {"daemon.idle_shutdown_s", nonNegative},
	"maintenance_gc_cron":           {"maintenance.gc_cron", cronSchedule},
	"placement_prefer":              {"placement.prefer", oneOf("fast", "balanced", "cheap")},
	"port":                          {"port", tcpPort},
	"yield":                         {"yield", oneOf("smart", "always", "never")},
}

func nonNegative(raw string) error {
	value, err := strconv.ParseInt(strings.TrimSpace(raw), 10, 64)
	if err != nil || value < 0 {
		return fmt.Errorf("is not a non-negative integer")
	}
	return nil
}

func boolean(raw string) error {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "true", "1", "yes", "false", "0", "no":
		return nil
	}
	return fmt.Errorf("is not true or false")
}

func cronSchedule(raw string) error {
	if expr := strings.TrimSpace(raw); expr != "" {
		if _, err := cron.ParseStandard(expr); err != nil {
			return fmt.Errorf("is not a five-field cron schedule")
		}
	}
	return nil
}

func tcpPort(raw string) error {
	value, err := strconv.Atoi(strings.TrimSpace(raw))
	if err != nil || value < 0 || value > 65535 {
		return fmt.Errorf("is not a TCP port or zero for automatic selection")
	}
	return nil
}

func oneOf(words ...string) func(string) error {
	return func(raw string) error {
		if slices.Contains(words, raw) {
			return nil
		}
		return fmt.Errorf("is not %s", strings.Join(words[:len(words)-1], ", ")+" or "+words[len(words)-1])
	}
}

type unusableSetting struct {
	name, spelled, raw, reason string
}

// admitBehaviour removes every unusable behaviour value from the file's settings so its
// default applies, and returns what it removed.
func admitBehaviour(file *resolver) []unusableSetting {
	var out []unusableSetting
	for name, setting := range behaviour {
		value, present := file.values[name]
		if !present {
			continue
		}
		raw := fmt.Sprint(value)
		reason := ""
		if file.conflicting[name] {
			reason = "is set more than once with different values"
		} else if err := setting.admit(raw); err != nil {
			reason = err.Error()
		}
		if reason != "" {
			delete(file.values, name)
			out = append(out, unusableSetting{name: name, spelled: setting.spelled, raw: raw, reason: reason})
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].spelled < out[j].spelled })
	return out
}

var fileKeys = map[string]bool{
	"tensorhub_url":                 true,
	"tensorhub_token":               true,
	"huggingface_token":             true,
	"civitai_token":                 true,
	"tfs":                           true,
	"tensorfs_root":                 true,
	"player_url":                    true,
	"local_rate_micro_usd_per_hour": true,
	"port":                          true,
	"yield":                         true,
	"rentals":                       true,
	"daemon":                        true,
	"maintenance":                   true,
	"placement":                     true,
	"hubs":                          true,
	"machine":                       true,
}

// nestedFileKeys are the one-level sections config.yaml admits, each mapping its
// nested spelling to the flat grammar name.
var nestedFileKeys = map[string]map[string]string{
	"rentals": {"development": "rentals_development",
		"ssh_public_key": "rentals_ssh_public_key"},
	"daemon":      {"idle_shutdown_s": "daemon_idle_shutdown_s"},
	"maintenance": {"gc_cron": "maintenance_gc_cron"},
	"placement":   {"prefer": "placement_prefer"},
	"machine":     {"gpu_budget": "machine_gpu_budget"},
}

var environmentNames = map[string]string{
	"tensorhub_url":     "TENSORHUB_URL",
	"tensorhub_token":   "TENSORHUB_TOKEN",
	"huggingface_token": "HF_TOKEN",
	"civitai_token":     "CIVITAI_TOKEN",
	"tfs":               "COZY_TFS",
	"tensorfs_root":     "TENSORFS_HOME",
	"tensorfs_registry": "COZY_TFS_REGISTRY",
}

var processConfig struct {
	once sync.Once
	cfg  Config
	err  *exit.Error
}

// Load resolves and freezes configuration. The first call owns the process
// snapshot; later calls return it.
func Load() (Config, *exit.Error) { return LoadForTensorhub(nil) }

// Terminal is what the environment says about the terminal, a cross-tool convention and not
// a Cozy setting: TERM=dumb cannot move the cursor, and a non-empty NO_COLOR asks for none.
type Terminal struct {
	Dumb, NoColor bool
}

// ReadTerminal is the terminal's environment facts; the entrypoint reads them once.
func ReadTerminal() Terminal {
	return Terminal{Dumb: os.Getenv("TERM") == "dumb", NoColor: os.Getenv("NO_COLOR") != ""}
}

// LoadForTensorhub freezes the explicit CLI selection (a hub name or URL) after file
// and environment resolution. Subsequent Load consumers see the same origin. The
// static token stays bound to the configured origin, so selecting another hub never
// carries it there.
func LoadForTensorhub(override *string) (Config, *exit.Error) {
	processConfig.once.Do(func() {
		processConfig.cfg, processConfig.err = load()
		if processConfig.err != nil || override == nil {
			return
		}
		origin, name, problem := ResolveHub(*override, processConfig.cfg.Hubs)
		if problem != nil {
			processConfig.err = exit.Named(exit.Validation, "config.tensorhub_url_invalid",
				"--tensorhub must name a hub (see `cozy hub list`) or be an http or https base URL without credentials, query, or fragment")
			return
		}
		processConfig.cfg = processConfig.cfg.ForHub(origin)
		processConfig.cfg.HubName, processConfig.cfg.HubURLSource = name, "flag"
	})
	return processConfig.cfg, processConfig.err
}

// ForHub is this configuration addressing another Tensorhub origin: the one view a
// record's own hub is served through. The static token is dropped unless it was
// issued for that origin; machine credentials are per origin already.
func (c Config) ForHub(origin string) Config {
	origin = strings.TrimRight(strings.TrimSpace(origin), "/")
	if origin == "" || origin == c.HubURL {
		return c
	}
	c.HubURL, c.HubName, c.HubURLSource = origin, c.hubNamed(origin), "record"
	c.HubToken, c.HubTokenSource = secret.New(""), "unset"
	if origin == c.ConfiguredHubURL && c.configuredToken.Present() {
		c.HubToken, c.HubTokenSource = c.configuredToken, c.configuredTokenSource
	}
	return c
}

// StaticToken reports whether an operator token is configured. It is bound to
// ConfiguredHubURL, so moving the current hub would silently carry it elsewhere.
func (c Config) StaticToken() bool { return c.configuredToken.Present() }

// StaticTokenSettings names where the operator token is set: the config.yaml key and
// the environment variable, each as the reader must remove it.
func (c Config) StaticTokenSettings() []string { return c.tokenSettings }

func (c Config) hubNamed(origin string) string {
	names := make([]string, 0, len(c.Hubs))
	for name, url := range c.Hubs {
		if url == origin {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	if len(names) == 0 {
		return ""
	}
	return names[0]
}

// HubLabel is how an origin is shown to a human: its hub name when it has one.
func (c Config) HubLabel(origin string) string {
	if name := c.hubNamed(strings.TrimRight(origin, "/")); name != "" {
		return name
	}
	return origin
}

// ResolveHub turns a hub name or URL into its origin and name.
func ResolveHub(value string, hubs map[string]string) (string, string, *exit.Error) {
	value = strings.TrimSpace(value)
	if origin, named := hubs[value]; named {
		return origin, value, nil
	}
	origin, problem := HubOrigin(value)
	if problem != nil {
		return "", "", problem
	}
	return origin, Config{Hubs: hubs}.hubNamed(origin), nil
}

// HubOrigin validates one Tensorhub base URL and returns its canonical spelling.
func HubOrigin(value string) (string, *exit.Error) {
	base := strings.TrimRight(strings.TrimSpace(value), "/")
	parsed, err := url.Parse(base)
	if err != nil || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") ||
		parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return "", exit.Named(exit.Validation, "config.tensorhub_url_invalid",
			"%q is not a hub name or an http or https Tensorhub base URL without credentials, query, or fragment", redactURL(value))
	}
	return base, nil
}

func redactURL(value string) string {
	if parsed, err := url.Parse(strings.TrimSpace(value)); err == nil && parsed.User != nil {
		parsed.User = nil
		return parsed.String()
	}
	return value
}

func load() (Config, *exit.Error) {
	home, problem := resolveHome()
	if problem != nil {
		return Config{}, problem
	}

	file, digest, problem := readConfigFile(filepath.Join(home, FileName))
	if problem != nil {
		return Config{}, problem
	}
	environment, inherited := readEnvironment()

	unusable := admitBehaviour(file)
	input, defaults, err := resolve(file, environment)
	if err != nil {
		return Config{}, exit.Usagef("configuration is invalid: %s", err).
			WithRemedy("check %s and the admitted COZY_/TENSORHUB_ environment values", filepath.Join(home, FileName))
	}
	for _, setting := range unusable {
		fmt.Fprintf(os.Stderr, "cozy: %s: %s %q %s; using %q\n",
			filepath.Join(home, FileName), setting.spelled, setting.raw, setting.reason, defaults[setting.name])
	}
	hubToken := secret.New(input.HubToken)
	huggingFaceToken := secret.New(input.HuggingFaceToken)
	civitaiToken := secret.New(input.CivitaiToken)
	tensorFSRoot, problem := resolveTensorFSRoot(input.TensorFSRoot)
	if problem != nil {
		return Config{}, problem
	}
	hubs := builtinHubs(file.hubs)
	hubURL, hubName, problem := ResolveHub(input.HubURL, hubs)
	if problem != nil {
		return Config{}, problem.WithRemedy("set tensorhub_url in %s to a name from `cozy hub list` or a Tensorhub URL", filepath.Join(home, FileName))
	}
	c := Config{
		Home:                     home,
		Port:                     input.Port,
		PortSource:               sourceOf("port", file, environment, "default"),
		Yield:                    input.Yield,
		HubURL:                   hubURL,
		HubName:                  hubName,
		Hubs:                     hubs,
		MachineGPUBudget:         file.gpuBudget,
		HubToken:                 hubToken,
		ConfiguredHubURL:         hubURL,
		HubURLSource:             sourceOf("tensorhub_url", file, environment, "default"),
		HubTokenSource:           sourceOf("tensorhub_token", file, environment, "unset"),
		HuggingFaceToken:         huggingFaceToken,
		HuggingFaceTokenSource:   sourceOf("huggingface_token", file, environment, "unset"),
		CivitaiToken:             civitaiToken,
		CivitaiTokenSource:       sourceOf("civitai_token", file, environment, "unset"),
		Tfs:                      strings.TrimSpace(input.Tfs),
		TfsSource:                sourceOf("tfs", file, environment, "default"),
		TensorFSRoot:             tensorFSRoot,
		TensorFSRootSource:       sourceOf("tensorfs_root", file, environment, "default"),
		TensorFSRegistry:         strings.TrimSpace(input.TensorFSRegistry),
		LocalRateMicroUSDPerHour: input.LocalRateMicroUSDPerHour,
		LocalRateSource:          sourceOf("local_rate_micro_usd_per_hour", file, environment, "unset"),
		RentalsDevelopment:       input.RentalsDevelopment,
		RentalsSSHPublicKey:      strings.TrimSpace(input.RentalsSSHPublicKey),
		DaemonIdleShutdown:       time.Duration(input.DaemonIdleShutdownS) * time.Second,
		MaintenanceGCCron:        strings.TrimSpace(input.MaintenanceGCCron),
		PlacementPrefer:          input.PlacementPrefer,
		PlayerURL:                strings.TrimSpace(input.PlayerURL),
		Digest:                   digest,
		inherited:                inherited,
	}
	if !hubToken.Present() {
		c.HubTokenSource = "unset"
	}
	c.configuredToken, c.configuredTokenSource = c.HubToken, c.HubTokenSource
	if hubToken.Present() {
		if file.has("tensorhub_token") {
			c.tokenSettings = append(c.tokenSettings, "the tensorhub_token line in "+filepath.Join(home, FileName))
		}
		if environment.has("tensorhub_token") {
			c.tokenSettings = append(c.tokenSettings, "the TENSORHUB_TOKEN environment variable")
		}
	}
	if !huggingFaceToken.Present() {
		c.HuggingFaceTokenSource = "unset"
	}
	if !civitaiToken.Present() {
		c.CivitaiTokenSource = "unset"
	}
	return c, nil
}

// resolve assigns the typed values and returns each setting's default beside them.
func resolve(file, environment *resolver) (values, map[string]string, error) {
	var input values
	parser, err := kong.New(&input,
		kong.Name("cozy-config"),
		kong.NoDefaultHelp(),
		kong.Resolvers(file, environment),
	)
	if err != nil {
		return values{}, nil, fmt.Errorf("cannot construct the configuration grammar: %w", err)
	}
	defaults := map[string]string{}
	for _, flag := range parser.Model.Flags {
		defaults[flag.Name] = flag.Default
	}
	if _, err := parser.Parse(nil); err != nil {
		return values{}, nil, err
	}
	return input, defaults, nil
}

func resolveHome() (string, *exit.Error) {
	if value := strings.TrimSpace(os.Getenv("COZY_HOME")); value != "" {
		home, err := filepath.Abs(value)
		if err != nil {
			return "", exit.Internalf("COZY_HOME %q is not resolvable: %s", value, err)
		}
		return home, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", exit.Internalf("no home directory and COZY_HOME is unset: %s", err)
	}
	return filepath.Join(home, ".cozy"), nil
}

// resolveTensorFSRoot turns the configured TensorFS root into an absolute path, or
// derives the product default `~/.tensorfs`. There is deliberately no `~/.cozy/cas`
// fallback and no compatibility lookup (proto-030).
func resolveTensorFSRoot(value string) (string, *exit.Error) {
	if value = strings.TrimSpace(value); value != "" {
		root, err := filepath.Abs(value)
		if err != nil {
			return "", exit.Internalf("tensorfs_root %q is not resolvable: %s", value, err)
		}
		return root, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", exit.Internalf("no home directory and no tensorfs_root configured: %s", err)
	}
	return filepath.Join(home, ".tensorfs"), nil
}

// readEnvironment captures the complete admitted environment once. Only
// credentials and external locations are resolvable configuration values;
// runtime behavior stays in config.yaml.
func readEnvironment() (*resolver, []string) {
	values := map[string]any{}
	for field, name := range environmentNames {
		if value := strings.TrimSpace(os.Getenv(name)); value != "" {
			values[field] = value
		}
	}

	inherited := make([]string, 0, len(Inherited))
	for _, name := range Inherited {
		if value, ok := os.LookupEnv(name); ok {
			inherited = append(inherited, name+"="+value)
		}
	}
	sort.Strings(inherited)
	return &resolver{values: values}, inherited
}

type resolver struct {
	values map[string]any
	// hubs is `hubs:`, the named Tensorhubs, name -> origin.
	hubs map[string]string
	// ignored names keys this build does not read: retired or newer settings.
	ignored []string
	// conflicting names behaviour settings the file gives two different values.
	conflicting map[string]bool
	// gpuBudget is `machine.gpu_budget`, verbatim: a size, a byte count, or a per-GPU map.
	gpuBudget any
}

func (r *resolver) Resolve(_ *kong.Context, _ *kong.Path, flag *kong.Flag) (any, error) {
	return r.values[flag.Name], nil
}

func (r *resolver) Validate(*kong.Application) error { return nil }

func (r *resolver) has(name string) bool {
	_, ok := r.values[name]
	return ok
}

func sourceOf(name string, file, environment *resolver, fallback string) string {
	switch {
	case environment.has(name):
		return "env"
	case file.has(name):
		return "file"
	default:
		return fallback
	}
}

func knownFileKeys() string {
	keys := make([]string, 0, len(fileKeys))
	for key := range fileKeys {
		if section, nested := nestedFileKeys[key]; nested {
			for name := range section {
				keys = append(keys, key+"."+name)
			}
			continue
		}
		if key == "hubs" {
			key = "hubs.<name>"
		}
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return strings.Join(keys, ", ")
}

func readConfigFile(path string) (*resolver, string, *exit.Error) {
	nameInfo, lstatErr := os.Lstat(path)
	if lstatErr != nil && !os.IsNotExist(lstatErr) {
		return nil, "", exit.Internalf("cannot inspect %s: %s", path, lstatErr)
	}
	file, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return &resolver{values: map[string]any{}}, digestOf(nil), nil
		}
		return nil, "", exit.Internalf("cannot read %s: %s", path, err)
	}
	defer file.Close()
	data, err := io.ReadAll(file)
	if err != nil {
		return nil, "", exit.Internalf("cannot read %s: %s", path, err)
	}

	resolved, err := fileYAML(bytes.NewReader(data))
	if err != nil {
		return nil, "", exit.Usagef("%s is invalid: %s", path, err).
			WithRemedy("this file admits: %s", knownFileKeys())
	}
	if len(resolved.ignored) > 0 {
		fmt.Fprintf(os.Stderr, "cozy: %s: this version does not use %s; ignored (it reads: %s)\n",
			path, strings.Join(resolved.ignored, ", "), knownFileKeys())
	}
	if resolved.has("huggingface_token") || resolved.has("civitai_token") {
		openedInfo, statErr := file.Stat()
		if statErr != nil {
			return nil, "", exit.Internalf("cannot inspect opened %s: %s", path, statErr)
		}
		if err := validateProviderSecretFile(nameInfo, openedInfo); err != nil {
			return nil, "", exit.Usagef("%s is invalid: %s", path, err).
				WithRemedy("store provider credentials in an owner-only regular file: chmod 600 %s", path)
		}
	}
	return resolved, digestOf(data), nil
}

func digestOf(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// builtinHubs adds the default hub's name unless the file names it itself.
func builtinHubs(hubs map[string]string) map[string]string {
	all := map[string]string{DefaultHubName: DefaultHubURL}
	for name, origin := range hubs {
		all[name] = origin
	}
	return all
}

var hubName = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]{0,62}$`)

// ValidHubName reports whether name can name a hub. A name is never a URL, so a
// selection is always unambiguous.
func ValidHubName(name string) bool { return hubName.MatchString(name) }

// hubsSection reads `hubs:`, the named Tensorhubs: each key a hub name, each value
// that hub's base URL.
func hubsSection(value *yaml.Node) (map[string]string, error) {
	if value.Kind != yaml.MappingNode {
		return nil, fmt.Errorf("line %d value for \"hubs\" is not a mapping of names to URLs", value.Line)
	}
	hubs := make(map[string]string, len(value.Content)/2)
	for j := 0; j < len(value.Content); j += 2 {
		name, url := value.Content[j], value.Content[j+1]
		if name.Kind != yaml.ScalarNode || !ValidHubName(name.Value) {
			return nil, fmt.Errorf("line %d hub name %q is not lowercase letters, digits, '.', '_' or '-'", name.Line, name.Value)
		}
		if _, exists := hubs[name.Value]; exists {
			return nil, fmt.Errorf("line %d names hub %q twice", name.Line, name.Value)
		}
		if url.Kind != yaml.ScalarNode {
			return nil, fmt.Errorf("line %d URL for hub %q is not a scalar", url.Line, name.Value)
		}
		origin, problem := HubOrigin(url.Value)
		if problem != nil {
			return nil, fmt.Errorf("line %d hub %q: %s", url.Line, name.Value, problem.Message)
		}
		hubs[name.Value] = origin
	}
	return hubs, nil
}

// fileYAML accepts one flat YAML mapping. Kong remains the typed assignment engine. A key
// this build does not read — retired by an upgrade or added by a newer build — is ignored
// and named once, so one stale line never disables every command. A location or
// credential repeated with different values refuses; a behaviour setting falls back.
func fileYAML(reader io.Reader) (*resolver, error) {
	decoder := yaml.NewDecoder(reader)
	var document yaml.Node
	if err := decoder.Decode(&document); err != nil {
		if err == io.EOF {
			return &resolver{values: map[string]any{}}, nil
		}
		return nil, err
	}
	var extra yaml.Node
	if err := decoder.Decode(&extra); err != io.EOF {
		if err == nil {
			return nil, fmt.Errorf("contains more than one YAML document")
		}
		return nil, err
	}
	if len(document.Content) == 0 {
		return &resolver{values: map[string]any{}}, nil
	}
	root := document.Content[0]
	if root.Kind != yaml.MappingNode {
		return nil, fmt.Errorf("must be one flat key-value mapping")
	}

	out := &resolver{values: make(map[string]any, len(root.Content)/2), conflicting: map[string]bool{}}
	set := func(name, spelled string, value *yaml.Node) error {
		if value.Kind != yaml.ScalarNode {
			return fmt.Errorf("line %d value for %q is not a scalar", value.Line, spelled)
		}
		if prior, exists := out.values[name]; exists && prior != value.Value {
			if _, tuning := behaviour[name]; tuning {
				out.conflicting[name] = true
				return nil
			}
			return fmt.Errorf("line %d names %q twice with different values", value.Line, spelled)
		}
		out.values[name] = value.Value
		return nil
	}
	for i := 0; i < len(root.Content); i += 2 {
		key, value := root.Content[i], root.Content[i+1]
		if key.Kind != yaml.ScalarNode || key.Value == "" {
			return nil, fmt.Errorf("line %d has a non-scalar or empty key", key.Line)
		}
		if !fileKeys[key.Value] {
			out.ignored = append(out.ignored, key.Value)
			continue
		}
		if key.Value == "machine" {
			if value.Kind != yaml.MappingNode {
				return nil, fmt.Errorf("line %d value for \"machine\" is not a mapping", value.Line)
			}
			for j := 0; j < len(value.Content); j += 2 {
				if value.Content[j].Value != "gpu_budget" {
					out.ignored = append(out.ignored, "machine."+value.Content[j].Value)
					continue
				}
				if err := value.Content[j+1].Decode(&out.gpuBudget); err != nil {
					return nil, fmt.Errorf("line %d machine.gpu_budget: %s", value.Line, err)
				}
			}
			continue
		}
		if key.Value == "hubs" {
			if out.hubs != nil {
				return nil, fmt.Errorf("line %d names %q twice", key.Line, key.Value)
			}
			hubs, err := hubsSection(value)
			if err != nil {
				return nil, err
			}
			out.hubs = hubs
			continue
		}
		if section, nested := nestedFileKeys[key.Value]; nested {
			if value.Kind != yaml.MappingNode {
				return nil, fmt.Errorf("line %d value for %q is not a mapping", value.Line, key.Value)
			}
			for j := 0; j < len(value.Content); j += 2 {
				nestedKey, nestedValue := value.Content[j], value.Content[j+1]
				spelled := key.Value + "." + nestedKey.Value
				name, ok := section[nestedKey.Value]
				if nestedKey.Kind != yaml.ScalarNode || !ok {
					out.ignored = append(out.ignored, spelled)
					continue
				}
				if err := set(name, spelled, nestedValue); err != nil {
					return nil, err
				}
			}
			continue
		}
		if err := set(key.Value, key.Value, value); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// Child constructs a child process's complete environment. Imposed values win
// over the captured allowlist and each variable appears once.
func (c Config) Child(imposed ...string) []string {
	if c.TfsSource != "default" && c.Tfs != "" {
		imposed = append([]string{"COZY_TFS=" + c.Tfs}, imposed...)
	}
	if c.TensorFSRootSource != "default" && c.TensorFSRoot != "" {
		imposed = append([]string{"TENSORFS_HOME=" + c.TensorFSRoot}, imposed...)
	}
	seen := map[string]string{}
	for _, pair := range c.inherited {
		name, value, _ := strings.Cut(pair, "=")
		seen[name] = value
	}
	for _, pair := range imposed {
		name, value, _ := strings.Cut(pair, "=")
		seen[name] = value
	}
	result := make([]string, 0, len(seen))
	for name, value := range seen {
		result = append(result, name+"="+value)
	}
	sort.Strings(result)
	return result
}

// Tool is Child plus the fixed no-color request used for parseable tool output.
func (c Config) Tool(imposed ...string) []string {
	return c.Child(append([]string{"NO_COLOR=1"}, imposed...)...)
}

// Frozen returns the one process snapshot for legacy consumers while cl-044
// moves them to explicit dependency injection.
func Frozen() Config { return processConfig.cfg }

// UseHub makes a hub name or URL the current hub (tensorhub_url) for later commands.
// Records keep their own hub, so switching never strands existing work.
func UseHub(home, selection string, hubs map[string]string) (string, *exit.Error) {
	origin, name, problem := ResolveHub(selection, hubs)
	if problem != nil {
		return "", problem
	}
	value := origin
	if name != "" {
		value = name
	}
	return origin, editFile(home, func(root *yaml.Node) bool {
		setScalar(root, "tensorhub_url", value)
		return true
	})
}

// AddHub names one Tensorhub URL (`hubs.<name>`), replacing an earlier URL for the name.
func AddHub(home, name, rawURL string) (string, *exit.Error) {
	if !ValidHubName(name) {
		return "", exit.Usagef("hub name %q is not lowercase letters, digits, '.', '_' or '-'", name)
	}
	origin, problem := HubOrigin(rawURL)
	if problem != nil {
		return "", problem
	}
	return origin, editFile(home, func(root *yaml.Node) bool {
		setScalar(mapping(root, "hubs"), name, origin)
		return true
	})
}

// RemoveHub forgets one hub name. The name config.yaml selects as current stays.
func RemoveHub(home, name string) *exit.Error {
	found, current := false, false
	problem := editFile(home, func(root *yaml.Node) bool {
		for i := 0; i < len(root.Content); i += 2 {
			if root.Content[i].Value == "tensorhub_url" && root.Content[i+1].Value == name {
				current = true
				return false
			}
		}
		section := mapping(root, "hubs")
		for i := 0; i < len(section.Content); i += 2 {
			if section.Content[i].Value == name {
				section.Content = append(section.Content[:i], section.Content[i+2:]...)
				found = true
				break
			}
		}
		if len(section.Content) == 0 {
			remove(root, "hubs")
		}
		return found
	})
	switch {
	case problem != nil:
		return problem
	case current:
		return exit.Named(exit.Conflict, "hub.current", "%s is the current hub", name).
			WithRemedy("switch first with `cozy hub use <other>`")
	case !found:
		return exit.Named(exit.NotFound, "hub.unknown", "config.yaml names no hub %q", name).
			WithNext("cozy hub list")
	}
	return nil
}

// editFile applies one edit to config.yaml, keeps everything else as written, refuses
// to write a file this package would not load, and replaces it atomically at 0600.
func editFile(home string, edit func(*yaml.Node) bool) *exit.Error {
	path := filepath.Join(home, FileName)
	data, err := os.ReadFile(path)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return exit.Internalf("cannot read %s: %s", path, err)
	}
	var document yaml.Node
	if len(bytes.TrimSpace(data)) > 0 {
		if err := yaml.Unmarshal(data, &document); err != nil {
			return exit.Usagef("%s is invalid: %s", path, err)
		}
	}
	if document.Kind == 0 {
		document = yaml.Node{Kind: yaml.DocumentNode, Content: []*yaml.Node{{Kind: yaml.MappingNode}}}
	}
	root := document.Content[0]
	if root.Kind != yaml.MappingNode {
		return exit.Usagef("%s is invalid: must be one flat key-value mapping", path)
	}
	if !edit(root) {
		return nil
	}
	var out bytes.Buffer
	encoder := yaml.NewEncoder(&out)
	encoder.SetIndent(2)
	if err := encoder.Encode(&document); err != nil {
		return exit.Internalf("cannot encode %s: %s", path, err)
	}
	_ = encoder.Close()
	if _, err := fileYAML(bytes.NewReader(out.Bytes())); err != nil {
		return exit.Usagef("%s would become invalid: %s", path, err)
	}
	if err := os.MkdirAll(home, 0o700); err != nil {
		return exit.Internalf("cannot create %s: %s", home, err)
	}
	temporary, err := os.CreateTemp(home, ".config-*.tmp")
	if err != nil {
		return exit.Internalf("cannot stage %s: %s", path, err)
	}
	defer os.Remove(temporary.Name())
	if err := temporary.Chmod(0o600); err == nil {
		_, err = temporary.Write(out.Bytes())
	}
	if err == nil {
		err = temporary.Sync()
	}
	if closeErr := temporary.Close(); err == nil {
		err = closeErr
	}
	if err == nil {
		err = os.Rename(temporary.Name(), path)
	}
	if err != nil {
		return exit.Internalf("cannot write %s: %s", path, err)
	}
	return nil
}

func mapping(root *yaml.Node, key string) *yaml.Node {
	for i := 0; i < len(root.Content); i += 2 {
		if root.Content[i].Value == key {
			if root.Content[i+1].Kind != yaml.MappingNode {
				root.Content[i+1] = &yaml.Node{Kind: yaml.MappingNode}
			}
			return root.Content[i+1]
		}
	}
	section := &yaml.Node{Kind: yaml.MappingNode}
	root.Content = append(root.Content, &yaml.Node{Kind: yaml.ScalarNode, Value: key}, section)
	return section
}

func setScalar(section *yaml.Node, key, value string) {
	for i := 0; i < len(section.Content); i += 2 {
		if section.Content[i].Value == key {
			section.Content[i+1] = &yaml.Node{Kind: yaml.ScalarNode, Value: value}
			return
		}
	}
	section.Content = append(section.Content, &yaml.Node{Kind: yaml.ScalarNode, Value: key},
		&yaml.Node{Kind: yaml.ScalarNode, Value: value})
}

func remove(root *yaml.Node, key string) {
	for i := 0; i < len(root.Content); i += 2 {
		if root.Content[i].Value == key {
			root.Content = append(root.Content[:i], root.Content[i+2:]...)
			return
		}
	}
}

// RuntimeYAML is the Runtime's runtime.yaml for a machine.gpu_budget value: gpu.budget,
// verbatim.
func RuntimeYAML(gpuBudget any) ([]byte, error) {
	return yaml.Marshal(map[string]any{"gpu": map[string]any{"budget": gpuBudget}})
}
