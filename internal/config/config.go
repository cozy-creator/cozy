// Package config owns Cozy Creator's one process-configuration read.
//
// Load resolves one private Kong grammar with no argv. Sources are, in order:
// defaults, $COZY_HOME/config.yaml, and process environment. The result is frozen for the process lifetime.
// Secrets therefore use Kong's resolver pipeline without becoming command-line
// flags.
package config

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/alecthomas/kong"
	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/secret"
	"go.yaml.in/yaml/v3"
)

// Inherited is the complete set of environment variables a package process
// may inherit. Launchers impose every other value explicitly.
var Inherited = []string{"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SSL_CERT_FILE", "SSL_CERT_DIR"}

const (
	DefaultHubURL = "http://127.0.0.1:8080"
	DefaultPort   = 0
	FileName      = "config.yaml"
)

// Config is the frozen value consumed by the rest of the process.
type Config struct {
	Home  string
	Port  int
	Yield string

	HubURL         string
	HubToken       secret.Value
	HubURLSource   string
	HubTokenSource string

	Tfs       string
	TfsSource string

	LocalRateMicroUSDPerHour int64
	LocalRateSource          string

	// Bootstrap is launcher-only. It is admitted from the process environment,
	// never config.yaml, argv, or a child inheritance list.
	Bootstrap secret.Value

	inherited []string
}

// values is a config-only Kong grammar. It is never embedded in the public CLI
// grammar and its parser always receives nil argv.
type values struct {
	HubURL                   string `name:"tensorhub_url" default:"http://127.0.0.1:8080"`
	HubToken                 string `name:"tensorhub_token"`
	Tfs                      string `name:"tfs" default:"tfs"`
	LocalRateMicroUSDPerHour int64  `name:"local_rate_micro_usd_per_hour" default:"0"`
	Port                     int    `name:"port" default:"0"`
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
	if v.Port < 0 || v.Port > 65535 {
		return fmt.Errorf("port %d is not a TCP port or zero for automatic selection", v.Port)
	}
	return nil
}

var fileKeys = map[string]bool{
	"tensorhub_url":                 true,
	"tensorhub_token":               true,
	"tfs":                           true,
	"local_rate_micro_usd_per_hour": true,
	"port":                          true,
	"yield":                         true,
}

var environmentNames = map[string]string{
	"tensorhub_url":   "TENSORHUB_URL",
	"tensorhub_token": "TENSORHUB_TOKEN",
	"tfs":             "COZY_TFS",
	"bootstrap":       "COZY_BOOTSTRAP_CREDENTIAL",
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

	file, problem := readConfigFile(filepath.Join(home, FileName))
	if problem != nil {
		return Config{}, problem
	}
	environment, inherited := readEnvironment()

	input, err := resolve(file, environment)
	if err != nil {
		return Config{}, exit.Usagef("configuration is invalid: %s", err).
			WithRemedy("check %s and the admitted COZY_/TENSORHUB_ environment values", filepath.Join(home, FileName))
	}

	hubToken := secret.New(input.HubToken)
	c := Config{
		Home:                     home,
		Port:                     input.Port,
		Yield:                    input.Yield,
		HubURL:                   strings.TrimRight(strings.TrimSpace(input.HubURL), "/"),
		HubToken:                 hubToken,
		HubURLSource:             sourceOf("tensorhub_url", file, environment, "default"),
		HubTokenSource:           sourceOf("tensorhub_token", file, environment, "unset"),
		Tfs:                      strings.TrimSpace(input.Tfs),
		TfsSource:                sourceOf("tfs", file, environment, "default"),
		LocalRateMicroUSDPerHour: input.LocalRateMicroUSDPerHour,
		LocalRateSource:          sourceOf("local_rate_micro_usd_per_hour", file, environment, "unset"),
		Bootstrap:                secret.New(input.Bootstrap),
		inherited:                inherited,
	}
	if !hubToken.Present() {
		c.HubTokenSource = "unset"
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
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return strings.Join(keys, ", ")
}

func readConfigFile(path string) (*resolver, *exit.Error) {
	file, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return &resolver{values: map[string]any{}}, nil
		}
		return nil, exit.Internalf("cannot read %s: %s", path, err)
	}
	defer file.Close()

	resolved, err := strictYAML(file)
	if err != nil {
		return nil, exit.Usagef("%s is invalid: %s", path, err).
			WithRemedy("this file admits: %s", knownFileKeys())
	}
	return resolved, nil
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
		if value.Kind != yaml.ScalarNode {
			return nil, fmt.Errorf("line %d value for %q is not a scalar", value.Line, key.Value)
		}
		values[key.Value] = value.Value
	}
	return &resolver{values: values}, nil
}

// Child constructs a child process's complete environment. Imposed values win
// over the captured allowlist and each variable appears once.
func (c Config) Child(imposed ...string) []string {
	if c.TfsSource != "default" && c.Tfs != "" {
		imposed = append([]string{"COZY_TFS=" + c.Tfs}, imposed...)
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
