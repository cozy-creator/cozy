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
	"fmt"
	"io"
	"os"
	"path/filepath"
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
	DefaultHubURL = "http://127.0.0.1:8819"
	DefaultPort   = 8818
	FileName      = "config.yaml"
)

// Config is the frozen value consumed by the rest of the process.
type Config struct {
	Home string
	Port int
	// PortSource distinguishes the product default, which may fall back when
	// occupied, from an operator-selected port that must bind exactly.
	PortSource string
	Yield      string

	HubURL         string
	HubToken       secret.Value
	HubURLSource   string
	HubTokenSource string

	HuggingFaceToken       secret.Value
	HuggingFaceTokenSource string
	CivitaiToken           secret.Value
	CivitaiTokenSource     string

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

	LocalRateMicroUSDPerHour       int64
	LocalRateSource                string
	RentalsMaxHourlySpendUSDMicros int64
	RentalsMaxHourlySpendSource    string
	// RentalsIdleRelease is how long a rented pod must have had no work — nothing queued
	// for it, nothing running or owed on it — before the daemon ends it, manual or
	// managed. A debounce over an observed fact, never the decision; zero leaves every
	// rental to `cozy rental end`.
	RentalsIdleRelease time.Duration
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

	// Bootstrap is launcher-only. It is admitted from the process environment,
	// never config.yaml, argv, or a child inheritance list.
	Bootstrap secret.Value

	inherited []string
}

// values is a config-only Kong grammar. It is never embedded in the public CLI
// grammar and its parser always receives nil argv.
type values struct {
	HubURL                   string `name:"tensorhub_url" default:"http://127.0.0.1:8819"`
	HubToken                 string `name:"tensorhub_token"`
	HuggingFaceToken         string `name:"huggingface_token"`
	CivitaiToken             string `name:"civitai_token"`
	Tfs                      string `name:"tfs" default:"tfs"`
	TensorFSRoot             string `name:"tensorfs_root"`
	TensorFSRegistry         string `name:"tensorfs_registry"`
	LocalRateMicroUSDPerHour int64  `name:"local_rate_micro_usd_per_hour" default:"0"`
	RentalsMaxHourlySpendUSD string `name:"rentals_max_hourly_spend_usd" default:"0"`
	RentalsIdleReleaseS      int64  `name:"rentals_idle_release_s" default:"300"`
	RentalsDevelopment       bool   `name:"rentals_development"`
	RentalsSSHPublicKey      string `name:"rentals_ssh_public_key"`
	DaemonIdleShutdownS      int64  `name:"daemon_idle_shutdown_s" default:"900"`
	MaintenanceGCCron        string `name:"maintenance_gc_cron" default:"0 3 * * *"`
	PlacementPrefer          string `name:"placement_prefer" default:"balanced"`
	Port                     int    `name:"port" default:"8818"`
	Yield                    string `name:"yield" default:"smart" enum:"smart,always,never"`
	Bootstrap                string `name:"bootstrap"`
}

func (v *values) Validate() error {
	if strings.TrimSpace(v.HubURL) == "" {
		return fmt.Errorf("tensorhub_url must not be empty")
	}
	if strings.TrimSpace(v.Tfs) == "" {
		return fmt.Errorf("tfs must not be empty")
	}
	if v.LocalRateMicroUSDPerHour < 0 {
		return fmt.Errorf("local_rate_micro_usd_per_hour must be non-negative")
	}
	if _, err := usdMicros(v.RentalsMaxHourlySpendUSD); err != nil {
		return fmt.Errorf("rentals.max_hourly_spend_usd %q is not a non-negative USD amount with at most six decimal places", v.RentalsMaxHourlySpendUSD)
	}
	if v.RentalsIdleReleaseS < 0 {
		return fmt.Errorf("rentals.idle_release_s must be non-negative; zero disables idle release")
	}
	if v.DaemonIdleShutdownS < 0 {
		return fmt.Errorf("daemon.idle_shutdown_s must be non-negative; zero disables idle shutdown")
	}
	if expr := strings.TrimSpace(v.MaintenanceGCCron); expr != "" {
		if _, err := cron.ParseStandard(expr); err != nil {
			return fmt.Errorf("maintenance.gc_cron %q is not a five-field cron schedule: %w", expr, err)
		}
	}
	switch v.PlacementPrefer {
	case "fast", "balanced", "cheap":
	default:
		return fmt.Errorf("placement.prefer %q is not fast, balanced or cheap", v.PlacementPrefer)
	}
	if v.Port < 0 || v.Port > 65535 {
		return fmt.Errorf("port %d is not a TCP port or zero for automatic selection", v.Port)
	}
	return nil
}

var fileKeys = map[string]bool{
	"tensorhub_url":                 true,
	"tensorhub_token":               true,
	"huggingface_token":             true,
	"civitai_token":                 true,
	"tfs":                           true,
	"tensorfs_root":                 true,
	"local_rate_micro_usd_per_hour": true,
	"port":                          true,
	"yield":                         true,
	"rentals":                       true,
	"daemon":                        true,
	"maintenance":                   true,
	"placement":                     true,
}

// nestedFileKeys are the one-level sections config.yaml admits, each mapping its
// nested spelling to the flat grammar name.
var nestedFileKeys = map[string]map[string]string{
	"rentals": {"max_hourly_spend_usd": "rentals_max_hourly_spend_usd",
		"idle_release_s": "rentals_idle_release_s", "development": "rentals_development",
		"ssh_public_key": "rentals_ssh_public_key"},
	"daemon":      {"idle_shutdown_s": "daemon_idle_shutdown_s"},
	"maintenance": {"gc_cron": "maintenance_gc_cron"},
	"placement":   {"prefer": "placement_prefer"},
}

var environmentNames = map[string]string{
	"tensorhub_url":     "TENSORHUB_URL",
	"tensorhub_token":   "TENSORHUB_TOKEN",
	"huggingface_token": "HF_TOKEN",
	"civitai_token":     "CIVITAI_TOKEN",
	"tfs":               "COZY_TFS",
	"tensorfs_root":     "TENSORFS_HOME",
	"tensorfs_registry": "COZY_TFS_REGISTRY",
	"bootstrap":         "COZY_BOOTSTRAP_CREDENTIAL",
}

var processConfig struct {
	once sync.Once
	cfg  Config
	err  *exit.Error
}

// Load resolves and freezes configuration. The first call owns the process
// snapshot; later calls return it.
func Load() (Config, *exit.Error) {
	processConfig.once.Do(func() {
		processConfig.cfg, processConfig.err = load()
	})
	return processConfig.cfg, processConfig.err
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

	input, err := resolve(file, environment)
	if err != nil {
		return Config{}, exit.Usagef("configuration is invalid: %s", err).
			WithRemedy("check %s and the admitted COZY_/TENSORHUB_ environment values", filepath.Join(home, FileName))
	}
	rentalCap, err := usdMicros(input.RentalsMaxHourlySpendUSD)
	if err != nil {
		return Config{}, exit.Usagef("configuration is invalid: %s", err)
	}

	hubToken := secret.New(input.HubToken)
	huggingFaceToken := secret.New(input.HuggingFaceToken)
	civitaiToken := secret.New(input.CivitaiToken)
	tensorFSRoot, problem := resolveTensorFSRoot(input.TensorFSRoot)
	if problem != nil {
		return Config{}, problem
	}
	c := Config{
		Home:                           home,
		Port:                           input.Port,
		PortSource:                     sourceOf("port", file, environment, "default"),
		Yield:                          input.Yield,
		HubURL:                         strings.TrimRight(strings.TrimSpace(input.HubURL), "/"),
		HubToken:                       hubToken,
		HubURLSource:                   sourceOf("tensorhub_url", file, environment, "default"),
		HubTokenSource:                 sourceOf("tensorhub_token", file, environment, "unset"),
		HuggingFaceToken:               huggingFaceToken,
		HuggingFaceTokenSource:         sourceOf("huggingface_token", file, environment, "unset"),
		CivitaiToken:                   civitaiToken,
		CivitaiTokenSource:             sourceOf("civitai_token", file, environment, "unset"),
		Tfs:                            strings.TrimSpace(input.Tfs),
		TfsSource:                      sourceOf("tfs", file, environment, "default"),
		TensorFSRoot:                   tensorFSRoot,
		TensorFSRootSource:             sourceOf("tensorfs_root", file, environment, "default"),
		TensorFSRegistry:               strings.TrimSpace(input.TensorFSRegistry),
		LocalRateMicroUSDPerHour:       input.LocalRateMicroUSDPerHour,
		LocalRateSource:                sourceOf("local_rate_micro_usd_per_hour", file, environment, "unset"),
		RentalsMaxHourlySpendUSDMicros: rentalCap,
		RentalsMaxHourlySpendSource:    sourceOf("rentals_max_hourly_spend_usd", file, environment, "unset"),
		RentalsIdleRelease:             time.Duration(input.RentalsIdleReleaseS) * time.Second,
		RentalsDevelopment:             input.RentalsDevelopment,
		RentalsSSHPublicKey:            strings.TrimSpace(input.RentalsSSHPublicKey),
		DaemonIdleShutdown:             time.Duration(input.DaemonIdleShutdownS) * time.Second,
		MaintenanceGCCron:              strings.TrimSpace(input.MaintenanceGCCron),
		PlacementPrefer:                input.PlacementPrefer,
		Digest:                         digest,
		Bootstrap:                      secret.New(input.Bootstrap),
		inherited:                      inherited,
	}
	if !hubToken.Present() {
		c.HubTokenSource = "unset"
	}
	if !huggingFaceToken.Present() {
		c.HuggingFaceTokenSource = "unset"
	}
	if !civitaiToken.Present() {
		c.CivitaiTokenSource = "unset"
	}
	return c, nil
}

func resolve(file, environment *resolver) (values, error) {
	var input values
	parser, err := kong.New(&input,
		kong.Name("cozy-config"),
		kong.NoDefaultHelp(),
		kong.Resolvers(file, environment),
	)
	if err != nil {
		return values{}, fmt.Errorf("cannot construct the configuration grammar: %w", err)
	}
	if _, err := parser.Parse(nil); err != nil {
		return values{}, err
	}
	return input, nil
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

type resolver struct{ values map[string]any }

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

	resolved, err := strictYAML(bytes.NewReader(data))
	if err != nil {
		return nil, "", exit.Usagef("%s is invalid: %s", path, err).
			WithRemedy("this file admits: %s", knownFileKeys())
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

// strictYAML accepts one flat YAML mapping. Kong remains the typed assignment
// and validation engine; this loader only protects the closed file vocabulary.
func strictYAML(reader io.Reader) (*resolver, error) {
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

	values := make(map[string]any, len(root.Content)/2)
	for i := 0; i < len(root.Content); i += 2 {
		key, value := root.Content[i], root.Content[i+1]
		if key.Kind != yaml.ScalarNode || key.Value == "" {
			return nil, fmt.Errorf("line %d has a non-scalar or empty key", key.Line)
		}
		if !fileKeys[key.Value] {
			return nil, fmt.Errorf("line %d names unknown key %q", key.Line, key.Value)
		}
		if _, exists := values[key.Value]; exists {
			return nil, fmt.Errorf("line %d names %q twice", key.Line, key.Value)
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
					return nil, fmt.Errorf("line %d names unknown key %q", nestedKey.Line, spelled)
				}
				if _, exists := values[name]; exists {
					return nil, fmt.Errorf("line %d names %q twice", nestedKey.Line, spelled)
				}
				if nestedValue.Kind != yaml.ScalarNode {
					return nil, fmt.Errorf("line %d value for %q is not a scalar", nestedValue.Line, spelled)
				}
				values[name] = nestedValue.Value
			}
			continue
		}
		if value.Kind != yaml.ScalarNode {
			return nil, fmt.Errorf("line %d value for %q is not a scalar", value.Line, key.Value)
		}
		values[key.Value] = value.Value
	}
	return &resolver{values: values}, nil
}

func usdMicros(value string) (int64, error) {
	value = strings.TrimSpace(value)
	whole, fraction, decimal := strings.Cut(value, ".")
	if whole == "" || strings.Contains(fraction, ".") || len(fraction) > 6 || decimal && fraction == "" {
		return 0, fmt.Errorf("invalid USD amount %q", value)
	}
	for _, part := range []string{whole, fraction} {
		for _, character := range part {
			if character < '0' || character > '9' {
				return 0, fmt.Errorf("invalid USD amount %q", value)
			}
		}
	}
	micros := strings.TrimLeft(whole+fraction+strings.Repeat("0", 6-len(fraction)), "0")
	if micros == "" {
		return 0, nil
	}
	amount, err := strconv.ParseInt(micros, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("invalid USD amount %q", value)
	}
	return amount, nil
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
