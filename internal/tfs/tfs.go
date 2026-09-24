// Package tfs is the ONE door to a byte-plane fact in this binary (cl-012).
//
// cozy-creator owns no byte plane: TensorFS does (README law 3, boundaries.md). So
// every question about canonical bytes — what a manifest reaches, what an object
// hashes to, whether the store holds it, whether a whole manifest verifies — is
// asked of the `tfs` binary and consumed as a document it produced or an identifier
// it printed. Nothing here decodes a manifest, a header, or a tensor.
//
// Machine seams return only ObjectRefs and repository rows owned by TensorFS. Cozy
// never decodes manifest/header/repository documents or composes store paths.
//
// Two fences hold this (scripts/fence.py, family `tensor`): no Go decoder for a
// canonical tensor document, and no second CAS key derivation — a path under
// `blobs/…` composed in Go would be a second spelling of TensorFS's own
// layout, and the two would drift.
package tfs

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hostruntime"
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
			WithRemedy("install tfs with the host Runtime: %s", hostruntime.InstallCommand)
	}
	t := &Tool{Bin: bin, Root: cfg.TensorFSRoot, Source: cfg.TfsSource, env: cfg.Tool()}
	if err := os.MkdirAll(t.Root, 0o755); err != nil {
		return nil, exit.Internalf("cannot create the local store at %s: %s", t.Root, err)
	}
	if _, e := t.run("store", "ensure", t.Root); e != nil {
		return nil, e
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
		for _, line := range strings.Split(said, "\n") {
			if line = strings.TrimSpace(line); strings.HasPrefix(line, "REFUSED ") {
				said = line
				break
			}
		}
		return out.String(), exit.Named(exit.Validation, "tfs_refused",
			"tfs %s: %s", strings.Join(redact(args), " "), firstLine(said)).
			WithRemedy("the byte plane refused in its own words; nothing above it may overrule that")
	}
	return out.String(), nil
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

// ---------------------------------------------------------------- identifiers

// Object is one entry of a publish closure: an identity and a length. This is the
// whole vocabulary cozy-creator needs about a byte — everything else is TensorFS's.
type Object struct {
	ID     string `json:"object_id"`
	Length int64  `json:"length"`
}

type refRow struct {
	Kind   string `json:"kind,omitempty"`
	Length int64  `json:"length"`
	Path   string `json:"path,omitempty"`
	SHA256 string `json:"sha256"`
}

func readRefs(path string) ([]refRow, *exit.Error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, exit.Internalf("the ObjectRefs tfs wrote are unreadable: %s", err)
	}
	var rows []refRow
	for line, text := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
		if strings.TrimSpace(text) == "" {
			continue
		}
		var row refRow
		if err := json.Unmarshal([]byte(text), &row); err != nil || len(row.SHA256) != 64 || row.Length <= 0 {
			return nil, exit.Internalf("tfs returned an invalid ObjectRef at line %d", line+1)
		}
		rows = append(rows, row)
	}
	return rows, nil
}

// ManifestObjects returns the sorted, distinct transitive blob closure. The
// manifest itself lives in its typed namespace and is not one of these blob refs.
func (t *Tool) ManifestObjects(manifestID, outPath string) ([]Object, *exit.Error) {
	if _, e := t.run("manifest", "walk", t.Root, hex(manifestID), "--refs", outPath); e != nil {
		return nil, e
	}
	rows, e := readRefs(outPath)
	if e != nil {
		return nil, e
	}
	objects := make([]Object, 0, len(rows))
	for _, row := range rows {
		objects = append(objects, Object{ID: "sha256:" + row.SHA256, Length: row.Length})
	}
	return objects, nil
}

// ManifestEntries inspects exact staged manifest bytes and returns direct refs.
// HeaderID is the one cozytensors entry; TensorFS owns that cardinality rule.
func (t *Tool) ManifestEntries(manifestPath, outPath string) (objects []Object, headerID string, e *exit.Error) {
	if _, e := t.run("manifest", "inspect", manifestPath, "--refs", outPath); e != nil {
		return nil, "", e
	}
	rows, e := readRefs(outPath)
	if e != nil {
		return nil, "", e
	}
	for _, row := range rows {
		objects = append(objects, Object{ID: "sha256:" + row.SHA256, Length: row.Length})
		if row.Kind == "cozytensors" {
			if headerID != "" {
				return nil, "", exit.Internalf("tfs returned more than one cozytensors entry")
			}
			headerID = "sha256:" + row.SHA256
		}
	}
	if headerID == "" {
		return nil, "", exit.Internalf("tfs returned no cozytensors entry")
	}
	return objects, headerID, nil
}

var present = regexp.MustCompile(`contains:\s+(true|false)`)

// Held answers whether the store already holds an object. It is a HINT and tfs says
// so on every line it prints: presence is never a resume predicate, which is why the
// resume path below is `Fill` (verification records) and not this.
func (t *Tool) Held(id string) (bool, *exit.Error) {
	out, e := t.run("contains", t.Root, hex(id))
	if e != nil {
		return false, e
	}
	m := present.FindStringSubmatch(out)
	return m != nil && m[1] == "true", nil
}

// ---------------------------------------------------------------- bytes

// Extract streams one verified object OUT of the store, for upload. The read is
// verified by tfs against the identity asked for, so a publisher cannot upload bytes
// its own store has silently corrupted.
func (t *Tool) Extract(id, outPath string) *exit.Error {
	_, e := t.run("get", t.Root, hex(id), "--out", outPath)
	return e
}

// Manifest writes exact, schema-checked bytes from the typed manifest namespace.
func (t *Tool) Manifest(id, outPath string) *exit.Error {
	_, e := t.run("manifest", "get", t.Root, hex(id), "--out", outPath)
	return e
}

var admittedManifest = regexp.MustCompile(`admitted=(true|false)`)

// AdmitManifest validates exact downloaded bytes and installs them in the typed
// manifest namespace. The bool says whether this call created the object.
func (t *Tool) AdmitManifest(path, id string, length int64) (bool, *exit.Error) {
	out, e := t.run("manifest", "admit", t.Root, path, "--expect", id,
		"--expect-length", strconv.FormatInt(length, 10))
	if e != nil {
		return false, e
	}
	m := admittedManifest.FindStringSubmatch(out)
	if m == nil {
		return false, exit.Internalf("tfs manifest admit printed no result")
	}
	return m[1] == "true", nil
}

// FillResult is what one resumable install pass did.
type FillResult struct {
	Put     int
	Skipped int
	Refused int
}

var fillLine = regexp.MustCompile(`fill: put (\d+), skipped (\d+), refused (\d+)`)

// Fill is the transactional install of a fetched set, and the whole resume story
// (cl-009's sibling on the CAS side): tfs skips only objects with a VALID
// verification record, rehashes a present-but-unverified one, quarantines a corrupt
// squatter out of the way, and admits every other object under its declared
// identity. The store's own records are the journal — this client keeps none.
func (t *Tool) Fill(plan []byte, planPath string) (FillResult, *exit.Error) {
	if err := os.WriteFile(planPath, plan, 0o644); err != nil {
		return FillResult{}, exit.Internalf("cannot write the fill plan: %s", err)
	}
	out, e := t.run("fill", t.Root, planPath)
	if e != nil {
		return FillResult{}, e
	}
	m := fillLine.FindStringSubmatch(out)
	if m == nil {
		return FillResult{}, exit.Internalf("tfs fill printed no result line")
	}
	put, _ := strconv.Atoi(m[1])
	skipped, _ := strconv.Atoi(m[2])
	refused, _ := strconv.Atoi(m[3])
	return FillResult{Put: put, Skipped: skipped, Refused: refused}, nil
}

// VerifyManifest hashes every blob reached by a manifest.
func (t *Tool) VerifyManifest(id string) *exit.Error {
	_, e := t.run("manifest", "verify", t.Root, hex(id))
	return e
}

// Release is one authoritative local repository row returned by TensorFS: a release
// lane (`kind: "release"`, with a version and lane) or a local alias (`kind: "local"`,
// with the source selection it was made from and no release at all).
type Release struct {
	Kind            string `json:"kind"`
	Org             string `json:"org"`
	Name            string `json:"name"`
	Version         string `json:"version"`
	Lane            string `json:"lane"`
	SourceSelection string `json:"source_selection"`
	ManifestSHA256  string `json:"manifest_sha256"`
	ManifestLength  int64  `json:"manifest_length"`
}

// Releases lists repository metadata through TensorFS, never through SQLite or
// direct traversal of repos/.
func (t *Tool) Releases(outPath string) ([]Release, *exit.Error) {
	if _, e := t.run("repo", "list", t.Root, "--rows", outPath); e != nil {
		return nil, e
	}
	raw, err := os.ReadFile(outPath)
	if err != nil {
		return nil, exit.Internalf("the repository rows tfs wrote are unreadable: %s", err)
	}
	var rows []Release
	for line, text := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
		if strings.TrimSpace(text) == "" {
			continue
		}
		var row Release
		if err := json.Unmarshal([]byte(text), &row); err != nil || row.Org == "" || row.Name == "" ||
			len(row.ManifestSHA256) != 64 || row.ManifestLength <= 0 {
			return nil, exit.Internalf("tfs returned an invalid repository row at line %d", line+1)
		}
		switch row.Kind {
		case "release":
			if row.Version == "" || row.Lane == "" {
				return nil, exit.Internalf("tfs returned a release row without a version and lane at line %d", line+1)
			}
		case "local":
			if row.Org != "local" || !validID(row.SourceSelection) || row.Version != "" || row.Lane != "" {
				return nil, exit.Internalf("tfs returned an invalid local alias row at line %d", line+1)
			}
		default:
			return nil, exit.Internalf("tfs returned a repository row of unknown kind %q at line %d", row.Kind, line+1)
		}
		rows = append(rows, row)
	}
	return rows, nil
}

// RepositoryUsage is one repository's blob bytes: everything its retained checkpoints
// reach, and the part no other repository reaches (what deleting only it would free).
type RepositoryUsage struct {
	Org    string `json:"org"`
	Name   string `json:"name"`
	Total  int64  `json:"bytes_total"`
	Unique int64  `json:"bytes_unique"`
}

// StoreUsage is the store's byte plane as TensorFS measures it: the union every
// repository reaches, the sum of their unique parts, and the verified blobs no retained
// manifest reaches.
type StoreUsage struct {
	Total        int64             `json:"bytes_total"`
	UniqueSum    int64             `json:"bytes_unique_sum"`
	Unreferenced int64             `json:"bytes_unreferenced"`
	Repos        []RepositoryUsage `json:"-"`
}

// Usage measures the store through TensorFS's own closure walk — the one its GC keeps
// alive — never by summing files under blobs/.
func (t *Tool) Usage(outPath string) (StoreUsage, *exit.Error) {
	if _, e := t.run("repo", "usage", t.Root, "--rows", outPath); e != nil {
		return StoreUsage{}, e
	}
	raw, err := os.ReadFile(outPath)
	if err != nil {
		return StoreUsage{}, exit.Internalf("the usage rows tfs wrote are unreadable: %s", err)
	}
	var usage StoreUsage
	store := false
	for line, text := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
		if strings.TrimSpace(text) == "" {
			continue
		}
		var kind struct {
			Kind  string `json:"kind"`
			Repos int    `json:"repos"`
		}
		if err := json.Unmarshal([]byte(text), &kind); err != nil {
			return StoreUsage{}, exit.Internalf("tfs returned an invalid usage row at line %d", line+1)
		}
		switch kind.Kind {
		case "repo":
			var row RepositoryUsage
			if err := json.Unmarshal([]byte(text), &row); err != nil || row.Org == "" || row.Name == "" ||
				row.Unique > row.Total {
				return StoreUsage{}, exit.Internalf("tfs returned an invalid usage row at line %d", line+1)
			}
			usage.Repos = append(usage.Repos, row)
		case "store":
			if store || json.Unmarshal([]byte(text), &usage) != nil || kind.Repos != len(usage.Repos) {
				return StoreUsage{}, exit.Internalf("tfs returned an invalid store usage row at line %d", line+1)
			}
			store = true
		default:
			return StoreUsage{}, exit.Internalf("tfs returned an unknown usage row kind %q at line %d", kind.Kind, line+1)
		}
	}
	if !store {
		return StoreUsage{}, exit.Internalf("tfs returned no store usage row")
	}
	return usage, nil
}

// LocalAlias is TensorFS's exact device-local alias projection. The repository
// digest is the compare-and-swap observation required by replace/remove.
type LocalAlias struct {
	ManifestDigest   string `json:"manifest_digest"`
	ManifestLength   int64  `json:"manifest_length"`
	Name             string `json:"name"`
	RepositoryDigest string `json:"repository_digest"`
	SourceSelection  string `json:"source_selection"`
}

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
	if len(members) > 4096 {
		return nil, exit.Internalf("tfs returned too many source members")
	}
	for index, member := range members {
		if member == "" || index > 0 && members[index-1] >= member {
			return nil, exit.Internalf("tfs returned unsorted or duplicate source members")
		}
	}
	return members, nil
}

// SameSelection ignores transient paths and sessions while binding the exact
// reviewed profile and source-to-component assignment.
func (p SourcePlan) SameSelection(other SourcePlan) bool {
	if p.Profile != other.Profile || p.RegistrySHA256 != other.RegistrySHA256 ||
		p.Target != other.Target || len(p.Sources) != len(other.Sources) {
		return false
	}
	for i := range p.Sources {
		left, right := p.Sources[i], other.Sources[i]
		if left.Component != right.Component || left.SourceMember != right.SourceMember ||
			left.Projected != right.Projected {
			return false
		}
	}
	return true
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

var candidateManifest = regexp.MustCompile(`(?m)^candidate\s+manifest\s+(sha256:[0-9a-f]{64})\s*$`)

func (t *Tool) RunSource(ctx context.Context, planPath string) (string, *exit.Error) {
	out, problem := t.runContext(ctx, "ingest", "run", t.Root, "--source-plan", planPath)
	if problem != nil {
		return "", problem
	}
	matches := candidateManifest.FindAllStringSubmatch(out, -1)
	if len(matches) != 1 {
		return "", exit.Internalf("tfs ingest run printed no unique candidate Manifest")
	}
	return matches[0][1], nil
}

// ObserveLocal returns the current compare-and-swap identity, or "absent".
func (t *Tool) ObserveLocal(name, outPath string) (string, *exit.Error) {
	rows, problem := t.Releases(outPath)
	if problem != nil {
		return "", problem
	}
	count := 0
	for _, row := range rows {
		if row.Org == "local" && row.Name == name {
			count++
		}
	}
	if count == 0 {
		return "absent", nil
	}
	if count != 1 {
		return "", exit.Named(exit.Conflict, "model.local_alias_ambiguous",
			"local/%s has %d releases instead of exactly one", name, count)
	}
	alias, problem := t.ResolveLocal(name)
	if problem != nil {
		return "", problem
	}
	return alias.RepositoryDigest, nil
}

func (t *Tool) InstallLocal(session, name, sourceSelection, observed string) *exit.Error {
	args := []string{"ingest", "install", t.Root, session, "local", name,
		strings.TrimPrefix(sourceSelection, "sha256:"), "local", "--observed", observed}
	_, problem := t.run(args...)
	return problem
}

// ReplaceLocal atomically points Creator's one local alias at an already verified
// WeightsSink Manifest. TensorFS owns manifest verification, repository bytes, and
// compare-and-swap; Creator supplies only frozen identities.
func (t *Tool) ReplaceLocal(name, sourceSelection, observed, manifestID string,
	manifestLength int64,
) (LocalAlias, *exit.Error) {
	args := []string{"local", "replace", t.Root, name, sourceSelection,
		manifestID, strconv.FormatInt(manifestLength, 10), "--observed", observed}
	out, problem := t.run(args...)
	if problem != nil {
		return LocalAlias{}, problem
	}
	return parseLocalAlias(out)
}

func parseLocalAlias(raw string) (LocalAlias, *exit.Error) {
	var alias LocalAlias
	if json.Unmarshal([]byte(strings.TrimSpace(raw)), &alias) != nil || alias.Name == "" ||
		alias.ManifestLength <= 0 || !validID(alias.ManifestDigest) ||
		!validID(alias.RepositoryDigest) || !validID(alias.SourceSelection) {
		return LocalAlias{}, exit.Internalf("tfs returned an invalid local alias projection")
	}
	return alias, nil
}

func validID(value string) bool {
	_, problem := ManifestID(value)
	return problem == nil
}

// ResolveLocal returns the one current row behind Creator's reserved local/name alias.
func (t *Tool) ResolveLocal(name string) (LocalAlias, *exit.Error) {
	out, e := t.run("local", "resolve", t.Root, name)
	if e != nil {
		return LocalAlias{}, e
	}
	return parseLocalAlias(out)
}

func (t *Tool) RemoveLocal(name, observed string) *exit.Error {
	_, e := t.run("local", "remove", t.Root, name, "--observed", observed)
	return e
}

func writeJSON(path string, value any) *exit.Error {
	raw, err := json.Marshal(value)
	if err != nil {
		return exit.Internalf("cannot encode the TensorFS request: %s", err)
	}
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		return exit.Internalf("cannot stage the TensorFS request: %s", err)
	}
	return nil
}

func (t *Tool) observedRepository(org, name, scratch string) (string, *exit.Error) {
	current := filepath.Join(scratch, "repo-current.json")
	if _, e := t.run("repo", "get", t.Root, org, name, "--out", current); e != nil {
		// The native CLI owns the distinction between an absent repository and
		// corrupt/unreadable metadata. Its release listing omits retained-only repos.
		if strings.Contains(e.Message, "REFUSED REPOSITORY_ABSENT:") {
			return "-", nil
		}
		return "", e
	}
	return current, nil
}

// RetainedCheckpoint reads the native repository's exact checkpoint membership.
func (t *Tool) RetainedCheckpoint(org, name, manifestID, scratch string) (int64, *exit.Error) {
	current, problem := t.observedRepository(org, name, scratch)
	if problem != nil || current == "-" {
		return 0, problem
	}
	raw, err := os.ReadFile(current)
	if err != nil {
		return 0, exit.Internalf("cannot read native repository metadata: %s", err)
	}
	var repository struct {
		Checkpoints []struct {
			Manifest struct {
				SHA256 string `json:"sha256"`
				Length int64  `json:"length"`
			} `json:"manifest"`
		} `json:"checkpoints"`
	}
	if err := json.Unmarshal(raw, &repository); err != nil {
		return 0, exit.Internalf("cannot decode native repository metadata: %s", err)
	}
	for _, checkpoint := range repository.Checkpoints {
		if "sha256:"+checkpoint.Manifest.SHA256 == manifestID {
			return checkpoint.Manifest.Length, nil
		}
	}
	return 0, nil
}

func releaseRevision(path, version string) (uint64, *exit.Error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return 0, exit.Internalf("the current TensorFS repository is unreadable: %s", err)
	}
	var repository struct {
		Releases []struct {
			Revision uint64 `json:"revision"`
			Version  string `json:"version"`
		} `json:"releases"`
	}
	if err := json.Unmarshal(raw, &repository); err != nil {
		return 0, exit.Internalf("the current TensorFS repository cannot be decoded: %s", err)
	}
	for _, release := range repository.Releases {
		if release.Version == version {
			if release.Revision == 0 {
				return 0, exit.Internalf("TensorFS returned release %s with revision zero", version)
			}
			return release.Revision, nil
		}
	}
	return 0, nil
}

// CommitCheckpoint retains a verified manifest without inventing a release.
func (t *Tool) CommitCheckpoint(org, name, manifestID string, length int64,
	scratch string,
) *exit.Error {
	current, e := t.observedRepository(org, name, scratch)
	if e != nil {
		return e
	}
	checkpointMutation := filepath.Join(scratch, "repo-put-checkpoint.json")
	if e := writeJSON(checkpointMutation, map[string]any{
		"action":   "put_checkpoint",
		"manifest": map[string]any{"length": length, "sha256": hex(manifestID)},
		"repo":     map[string]string{"name": name, "org": org},
	}); e != nil {
		return e
	}
	if _, e = t.run("repo", "commit", t.Root, current, checkpointMutation); e != nil {
		return e
	}
	return nil
}

// CommitRelease names an already verified and retained checkpoint locally.
func (t *Tool) CommitRelease(org, name, version, lane, manifestID string, length int64,
	scratch string,
) *exit.Error {
	if e := t.CommitCheckpoint(org, name, manifestID, length, scratch); e != nil {
		return e
	}
	checkpointed := filepath.Join(scratch, "repo-checkpointed.json")
	if _, e := t.run("repo", "get", t.Root, org, name, "--out", checkpointed); e != nil {
		return e
	}
	expectedRevision, e := releaseRevision(checkpointed, version)
	if e != nil {
		return e
	}
	releaseMutation := filepath.Join(scratch, "repo-update-release.json")
	if e := writeJSON(releaseMutation, map[string]any{
		"action": "update_release", "expected_revision": expectedRevision,
		"remove": []string{}, "repo": map[string]string{"name": name, "org": org},
		"set": []map[string]any{{"lane": lane,
			"manifest": map[string]any{"length": length, "sha256": hex(manifestID)}}},
		"version": version,
	}); e != nil {
		return e
	}
	_, e = t.run("repo", "commit", t.Root, checkpointed, releaseMutation)
	return e
}

// GCReport is one reclamation pass as TensorFS reports it (`tfs gc --json`).
type GCReport struct {
	DryRun             bool     `json:"dry_run"`
	ReclaimedBytes     int64    `json:"reclaimed_bytes"`
	ReclaimedBlobs     int64    `json:"reclaimed_blobs"`
	ReclaimedManifests int64    `json:"reclaimed_manifests"`
	KeptBytes          int64    `json:"kept_bytes"`
	KeptObjects        int64    `json:"kept_objects"`
	Sessions           []string `json:"sessions"`
	ScratchReaped      int64    `json:"scratch_reaped"`
}

// GC is the reclamation act over the local store (owner ruling 2026-09-02): TensorFS
// removes every blob and manifest no repository names, deciding from its own filesystem
// census — repos, manifests, blobs — never from a database. A live writer or read lease
// refuses by name; an open ingest session keeps what it names. `abandonSessions` first
// abandons every session whose writer is gone (`tfs ingest reap`); a caller says so only
// when none of its own transfers can still stand between `ingest run` and `ingest install`.
func (t *Tool) GC(abandonSessions bool) (GCReport, *exit.Error) {
	if abandonSessions {
		if _, problem := t.run("ingest", "reap", t.Root); problem != nil {
			return GCReport{}, problem
		}
	}
	out, problem := t.run("gc", t.Root, "--json")
	if problem != nil {
		return GCReport{}, problem
	}
	var report GCReport
	if err := json.Unmarshal([]byte(strings.TrimSpace(out)), &report); err != nil {
		return GCReport{}, exit.Internalf("tfs gc returned an unreadable report: %s", err)
	}
	return report, nil
}

// DeleteRepository removes the local durable name. Its bytes are reclaimed by the GC that
// follows the removal.
func (t *Tool) DeleteRepository(org, name, scratch string) *exit.Error {
	current, e := t.observedRepository(org, name, scratch)
	if e != nil {
		return e
	}
	if current == "-" {
		return nil
	}
	mutation := filepath.Join(scratch, "repo-delete.json")
	if e := writeJSON(mutation, map[string]any{
		"action": "delete_repository", "repo": map[string]string{"name": name, "org": org},
	}); e != nil {
		return e
	}
	_, e = t.run("repo", "commit", t.Root, current, mutation)
	return e
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
