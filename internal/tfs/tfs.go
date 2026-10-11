// Package tfs runs the `tfs` binary for what this computer still asks TensorFS directly:
// opening the store and planning a provider source's conversion from its headers before
// anything is rented. Machines hold and move model bytes; nothing here decodes a manifest,
// a header or a tensor.
package tfs

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
)

// Tool is one resolved tensorfs CLI plus the store it operates on.
type Tool struct {
	Bin    string
	Root   string // the local CAS
	Source string // where the binary path came from, for the remedy text
	env    []string
}

// Open resolves the binary and the independent TensorFS Store root (proto-030: the
// root is supplied explicitly, never derived from the Creator home). A missing `tfs`
// is a structural refusal (exit 6) naming what to install: this binary cannot
// substitute for it, and a cozy-creator-side reimplementation is exactly what
// boundaries.md forbids.
func Open(cfg config.Config) (*Tool, *exit.Error) {
	bin, err := exec.LookPath(cfg.Tfs)
	if err != nil {
		return nil, exit.Named(exit.Structural, "tfs_missing",
			"the tensorfs CLI %q is not on PATH: %s", cfg.Tfs, err).
			WithRemedy("this command still reads a store on this computer and needs tfs, which cozy no longer installs; run it with --rental, or put a tensorfs release's tfs on PATH")
	}
	t := &Tool{Bin: bin, Root: cfg.TensorFSRoot, Source: cfg.TfsSource, env: cfg.Tool()}
	if err := os.MkdirAll(t.Root, 0o755); err != nil {
		return nil, exit.Internalf("cannot create the local store at %s: %s", t.Root, err)
	}
	// A Store is created only where none is: an existing one is opened by each verb as it is
	// (TensorFS 0.4 removed `store ensure`, which did both).
	if entries, err := os.ReadDir(t.Root); err == nil && len(entries) == 0 {
		if _, e := t.run("store", "init", t.Root); e != nil {
			return nil, e
		}
	}
	return t, nil
}

func (t *Tool) run(args ...string) (string, *exit.Error) {
	return t.runContext(context.Background(), args...)
}

func (t *Tool) runContext(ctx context.Context, args ...string) (string, *exit.Error) {
	cmd := exec.CommandContext(ctx, t.Bin, args...)
	cmd.Env = t.env
	var out, errb bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &errb
	if err := cmd.Run(); err != nil {
		// tfs refuses in its own vocabulary (REFUSED <CODE>: …). That vocabulary is
		// the byte plane's, so it reaches the user unchanged; only the exit code is ours.
		said := strings.TrimSpace(errb.String())
		if said == "" {
			said = strings.TrimSpace(out.String())
		}
		if said == "" {
			said = err.Error()
		}
		// Namespace/tool warnings can precede the actual refusal. Preserve the
		// byte plane's typed cause instead of attributing failure to a warning.
		code := ""
		for _, line := range strings.Split(said, "\n") {
			if line = strings.TrimSpace(line); strings.HasPrefix(line, "REFUSED ") {
				said = line
				code, _, _ = strings.Cut(strings.TrimPrefix(line, "REFUSED "), ":")
				break
			}
		}
		return out.String(), exit.Named(exit.Validation, "tfs_refused",
			"tfs %s: %s", strings.Join(redact(args), " "), firstLine(said)).
			WithRemedy("the byte plane refused in its own words; nothing above it may overrule that").
			WithCause(code)
	}
	return out.String(), nil
}

// Refused reports whether problem is tfs refusing with its own code.
func Refused(problem *exit.Error, code string) bool {
	return problem != nil && problem.Name == "tfs_refused" && problem.Cause == code
}

// Version is this host's `tfs version` line, for notes that name which TensorFS spoke.
func (t *Tool) Version() string {
	out, problem := t.run("version")
	if fields := strings.Fields(out); problem == nil && len(fields) >= 2 {
		return fields[0] + " " + fields[1]
	}
	return "tfs (version unknown)"
}

// redact keeps an absolute store path out of a refusal line's head, where it buries
// the reason under a path nobody is debugging.
func redact(args []string) []string {
	out := make([]string, 0, len(args))
	for _, a := range args {
		if filepath.IsAbs(a) {
			out = append(out, filepath.Base(a))
			continue
		}
		out = append(out, a)
	}
	return out
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}

// AsIsProfile is TensorFS's profile for a source no reviewed profile recognizes: its keys are
// stored as they are, and a package may need to normalize them.
const (
	AsIsProfile = "as-is/1"
	AsIsNote    = "unrecognized layout; stored as-is; packages may need to normalize it"
)

type SourceCarrier struct {
	Member string
	Path   string
}

type SourcePlan struct {
	Session        string `json:"session"`
	Profile        string `json:"profile"`
	RegistrySHA256 string `json:"registry_sha256"`
	Sources        []struct {
		Component    string `json:"component"`
		Path         string `json:"path"`
		SourceMember string `json:"source_member,omitempty"`
		Projected    bool   `json:"projected"`
	} `json:"sources"`
	Target string `json:"target"`
}

// SourceProfileMembers asks TensorFS which exact provider carriers a set of
// reviewed profiles names. It is a registry-only query: no model body or
// tensor header is opened.
func (t *Tool) SourceProfileMembers(registry string, profiles []string) ([]string, *exit.Error) {
	profiles = append([]string(nil), profiles...)
	sort.Strings(profiles)
	args := append([]string{"ingest", "source-members"}, profiles...)
	if registry != "" {
		args = append(args, "--registry", registry)
	}
	out, problem := t.run(args...)
	if problem != nil {
		return nil, problem
	}
	out = strings.TrimSuffix(out, "\n")
	if out == "" || strings.Contains(out, "\r") {
		return nil, exit.Internalf("tfs returned an invalid source-member selection")
	}
	members := strings.Split(out, "\n")
	if len(members) > 4096 || slices.Contains(members, "") {
		return nil, exit.Internalf("tfs returned an invalid source-member selection")
	}
	slices.Sort(members)
	return slices.Compact(members), nil
}

// PlanSource asks TensorFS to assign exact physical carriers through its
// reviewed whole-source profiles. Creator never parses tensor headers or maps
// filenames to components.
func (t *Tool) PlanSource(registry string, carriers []SourceCarrier, outPath string) (SourcePlan, *exit.Error) {
	return t.planSource(registry, "", carriers, outPath)
}

func (t *Tool) PlanSourceProfile(registry, profile string, carriers []SourceCarrier,
	outPath string,
) (SourcePlan, *exit.Error) {
	if strings.TrimSpace(profile) == "" {
		return SourcePlan{}, exit.New(exit.Validation, "source profile is empty")
	}
	return t.planSource(registry, profile, carriers, outPath)
}

func (t *Tool) planSource(registry, profile string, carriers []SourceCarrier,
	outPath string,
) (SourcePlan, *exit.Error) {
	args := []string{"ingest", "source-plan", "--out", outPath}
	if registry != "" {
		args = append(args, "--registry", registry)
	}
	if profile != "" {
		args = append(args, "--source-profile", profile)
	}
	for _, carrier := range carriers {
		value := carrier.Path
		if carrier.Member != "" {
			value = carrier.Member + "=" + carrier.Path
		}
		args = append(args, "--carrier", value)
	}
	if _, problem := t.run(args...); problem != nil {
		return SourcePlan{}, problem
	}
	raw, err := os.ReadFile(outPath)
	if err != nil {
		return SourcePlan{}, exit.Internalf("the source plan tfs wrote is unreadable: %s", err)
	}
	var plan SourcePlan
	if json.Unmarshal(raw, &plan) != nil || len(plan.Session) != 16 || plan.Profile == "" ||
		!validID(plan.RegistrySHA256) || plan.Target == "" || len(plan.Sources) == 0 {
		return SourcePlan{}, exit.Internalf("tfs returned an invalid source plan")
	}
	for _, source := range plan.Sources {
		if source.Component == "" || source.Path == "" {
			return SourcePlan{}, exit.Internalf("tfs returned an incomplete source mapping")
		}
	}
	return plan, nil
}

func (t *Tool) PreviewSource(planPath string) *exit.Error {
	_, problem := t.run("ingest", "plan", "--source-plan", planPath)
	return problem
}

func validID(value string) bool {
	_, problem := ManifestID(value)
	return problem == nil
}

// hex strips the `sha256:` spelling for the argument position tfs takes bare hex in.
func hex(id string) string { return strings.TrimPrefix(id, "sha256:") }

// ManifestID normalizes a manifest reference to the `sha256:<64 hex>` spelling.
func ManifestID(ref string) (string, *exit.Error) {
	h := hex(strings.TrimSpace(ref))
	if len(h) != 64 {
		return "", exit.Usagef("%q is not a manifest id", ref).
			WithRemedy("a manifest id is sha256:<64 hex> — TensorFS ingest and `cozy model download` print one")
	}
	for _, c := range h {
		if !strings.ContainsRune("0123456789abcdef", c) {
			return "", exit.Usagef("%q is not a manifest id: %q is not a hex digit", ref, c).
				WithRemedy("a manifest id is sha256:<64 hex>, lowercase")
		}
	}
	return "sha256:" + h, nil
}
