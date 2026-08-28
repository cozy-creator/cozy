package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	pb "github.com/cozy-creator/cozy-creator-v2/protocol/cozy/worker/v1"
)

// Release fixtures are DERIVED from the current tree (#616.c, cl-030). An archive on disk
// is at most a cache: beside it sits a derivation record naming the inputs it was built
// from — the wire schema digest, the runtime SHA, the creator-tree sources, the builder
// script itself — and when any of that mismatches the current tree the harness rebuilds
// the archive from source and proceeds. It never refuses pending a manual rebuild: the
// rev-4 wire bump left three rev-3 archives each refusing `wire_schema_mismatch` at Claim
// until a human ran a script the harness could have run itself. Packaging is CPU work
// (venv build, descriptor, tar) — deriving an archive never needs a GPU even where
// RUNNING its endpoint does.
type fixtureSpec struct {
	name string
	// over is the explicit archive override flag. Given, the human pointed at bytes;
	// they are used as-is and no derivation is consulted.
	over    string
	script  string
	args    []string
	slug    string
	sources []string
}

var sdxlUnetFixture = fixtureSpec{
	name: "sdxl-unet", over: "release", slug: "sdxl-unet",
	script: "scripts/sdxl-release.sh", args: []string{"--kind", "unet"},
}

var censusFixture = fixtureSpec{
	name: "census", over: "job-release", slug: "census",
	script: "scripts/job-release.sh",
}

var transportFixture = fixtureSpec{
	name: "weightless-transport", over: "transport-release", slug: "weightless",
	script:  "scripts/weightless-release.sh",
	sources: []string{"fixtures/weightless/weightless.py", "fixtures/weightless/endpoint.toml"},
}

// pipelineFixtureFor is both rungs of the four-component pipeline. The endpoint source is
// byte-identical across rungs; the fp8 release differs only in the `[bindings]` rung its
// endpoint.toml selects, which is the two-sided contract's point.
func pipelineFixtureFor(slug string) fixtureSpec {
	fx := fixtureSpec{
		name: slug, over: "pipeline-release-" + slug, slug: slug,
		script:  "scripts/sdxl-release.sh",
		args:    []string{"--kind", "pipeline"},
		sources: []string{"scripts/sdxl_proof.py"},
	}
	if slug != "sdxl-pipeline" {
		rung := strings.TrimPrefix(slug, "sdxl-pipeline-")
		fx.args = append(fx.args, "--endpoint", "cozy/"+slug, "--artifact-release", rung)
	}
	return fx
}

// derivation is the recorded input set, written beside the archive it derives.
type derivation struct {
	WireSchemaDigest string            `json:"wire_schema_digest"`
	RuntimeSHA       string            `json:"runtime_sha"`
	ScriptSHA256     map[string]string `json:"script_sha256"`
	SourceSHA256     map[string]string `json:"source_sha256"`
	ArchiveSHA256    string            `json:"archive_sha256"`
	BuiltAt          string            `json:"built_at"`
}

// deriveFixture returns a release archive matching the current tree, rebuilding it first
// when the cached one's recorded derivation mismatches. The returned path always exists.
func deriveFixture(fx fixtureSpec) string {
	if v, ok := flags[fx.over]; ok && v != "" {
		if _, err := os.Stat(v); err != nil {
			must("the "+fx.name+" archive override", err)
		}
		return v
	}
	home, err := os.UserHomeDir()
	must("the home directory", err)
	dir := filepath.Join(home, ".cache", "cozy", "live-fixtures", fx.name)
	archive := filepath.Join(dir, fx.slug+"-1.0.0.tar.gz")
	sidecar := archive + ".derivation.json"

	wanted := wantedDerivation(fx)
	if reason := staleness(archive, sidecar, wanted); reason != "" {
		fmt.Printf("  fixture %s: rebuilding from the current tree — %s\n", fx.name, reason)
		rebuildFixture(fx, dir, archive, sidecar, wanted)
	}
	return archive
}

// staleness names the first way the cached archive's identity departs from the current
// tree, or "" for a cache that is exactly the derivation the tree wants.
func staleness(archive, sidecar string, wanted derivation) string {
	data, err := os.ReadFile(sidecar)
	if err != nil {
		return "no derivation record"
	}
	got := derivation{}
	if err := json.Unmarshal(data, &got); err != nil {
		return "unreadable derivation record: " + err.Error()
	}
	if got.WireSchemaDigest != wanted.WireSchemaDigest {
		return fmt.Sprintf("built against wire schema %s; this tree speaks %s",
			shortDigest(got.WireSchemaDigest), shortDigest(wanted.WireSchemaDigest))
	}
	// The record could agree while the runtime it names does not (a hand-edited record, or
	// a wire_identity rewritten in place at that SHA's spelling). The SHA's own declaration
	// is the fact; the record is its cache.
	if declared := runtimeDeclares(got.RuntimeSHA); declared != wanted.WireSchemaDigest {
		return fmt.Sprintf("recorded runtime %s declares wire schema %s, not this tree's %s",
			shortCommit(got.RuntimeSHA), shortDigest(declared), shortDigest(wanted.WireSchemaDigest))
	}
	if got.RuntimeSHA != wanted.RuntimeSHA {
		return fmt.Sprintf("built against runtime %s; current compatible runtime is %s",
			shortCommit(got.RuntimeSHA), shortCommit(wanted.RuntimeSHA))
	}
	for script, want := range wanted.ScriptSHA256 {
		if got.ScriptSHA256[script] != want {
			return "the build recipe " + script + " changed"
		}
	}
	for src, want := range wanted.SourceSHA256 {
		if got.SourceSHA256[src] != want {
			return "the source " + src + " changed"
		}
	}
	sum, err := sha256File(archive)
	if err != nil {
		return "no cached archive"
	}
	if sum != got.ArchiveSHA256 {
		return "the cached archive is not the bytes its record derives"
	}
	return ""
}

// rebuildFixture runs the builder against a runtime SHA that speaks this tree's wire
// schema, then records the derivation the archive now embodies.
func rebuildFixture(fx fixtureSpec, dir, archive, sidecar string, wanted derivation) {
	sha := resolveRuntimeSHA()
	must("creating the fixture cache", os.MkdirAll(dir, 0o755))
	script, err := filepath.Abs(filepath.Join(treeRoot(), fx.script))
	must("resolving "+fx.script, err)
	args := append([]string{script}, fx.args...)
	args = append(args, "--runtime-sha", sha, "--out", dir)
	must("naming the runtime repo for the builder", os.Setenv("RUNTIME_REPO", runtimeRepo()))
	cmd := niceCmd("bash", args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Println(string(out))
		must("deriving the "+fx.name+" archive", err)
	}
	sum, err := sha256File(archive)
	must("hashing the derived "+fx.name+" archive", err)

	wanted.RuntimeSHA = sha
	wanted.ArchiveSHA256 = sum
	wanted.BuiltAt = time.Now().UTC().Format(time.RFC3339)
	record, err := json.MarshalIndent(wanted, "", "  ")
	must("rendering the derivation record", err)
	must("recording the derivation", os.WriteFile(sidecar, append(record, '\n'), 0o644))
	fmt.Printf("  fixture %s: derived at runtime %s, %s\n",
		fx.name, shortCommit(sha), shortDigest("sha256:"+sum))
}

// resolveRuntimeSHA picks the runtime commit a rebuild packages: `--runtime-sha` when the
// driver was given one, else the runtime repo's HEAD — and in either case the SHA must
// itself declare this tree's wire schema. A candidate that does not is not a stale cache;
// it is two repositories at incompatible revisions, and no archive built from it could be
// anything but the refusal this mechanism deletes.
func resolveRuntimeSHA() string {
	candidate := flag("runtime-sha", "")
	if candidate == "" {
		out, err := runOut("git", "-C", runtimeRepo(), "rev-parse", "HEAD")
		must("resolving the runtime repo's HEAD", err)
		candidate = strings.TrimSpace(out)
	}
	if declared := runtimeDeclares(candidate); declared != pb.SchemaDigest {
		must("a runtime speaking this tree's wire schema", fmt.Errorf(
			"runtime %s declares %s and this build speaks %s — regenerate one side from "+
				"the other's worker-protocol revision (this build is rev %d)",
			shortCommit(candidate), shortDigest(declared), shortDigest(pb.SchemaDigest),
			pb.WireSchemaRev))
	}
	return candidate
}

var schemaDigestLine = regexp.MustCompile(`SCHEMA_DIGEST = "([^"]+)"`)

// runtimeDeclares reads the wire schema digest a runtime commit's own generated binding
// declares — from the pinned read-only object, never a working tree with its concurrent
// writer. "" is a commit with no declaration at all: one older than the fence itself.
func runtimeDeclares(sha string) string {
	if sha == "" {
		return ""
	}
	out, err := runOut("git", "-C", runtimeRepo(), "show",
		sha+":src/cozy/worker/v1/wire_identity.py")
	if err != nil {
		return ""
	}
	if m := schemaDigestLine.FindStringSubmatch(out); m != nil {
		return m[1]
	}
	return ""
}

func runtimeRepo() string {
	return flag("runtime", "/home/fidika/cozy_v2/cozy-runtime")
}

// treeRoot is the creator tree the fixtures derive from — the directory the driver runs
// in, which is where the product binary lives too.
func treeRoot() string {
	root, err := filepath.Abs(flag("repo", "."))
	must("resolving the creator tree", err)
	return root
}

func hashTreeFile(rel string) string {
	sum, err := sha256File(filepath.Join(treeRoot(), rel))
	must("hashing "+rel, err)
	return sum
}

func sha256File(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return fmt.Sprintf("%x", h.Sum(nil)), nil
}

func shortDigest(d string) string {
	if hex, ok := strings.CutPrefix(d, "sha256:"); ok && len(hex) >= 8 {
		return "sha256:" + hex[:8]
	}
	if d == "" {
		return "nothing"
	}
	return d
}

func shortCommit(sha string) string {
	if len(sha) >= 12 {
		return sha[:12]
	}
	if sha == "" {
		return "(none)"
	}
	return sha
}

// sectionFixtures derives every archive the sections consume, on CPU alone. This is the
// packaging half of every GPU-gated section run early: after it, no section can be wedged
// by a stored fixture, because there are none — only caches this section just proved the
// tree can re-derive.
func sectionFixtures() {
	head("every consumed fixture derives from the current tree")
	for _, fx := range []fixtureSpec{
		transportFixture,
		censusFixture,
		sdxlUnetFixture,
		pipelineFixtureFor("sdxl-pipeline"),
		pipelineFixtureFor("sdxl-pipeline-fp8"),
	} {
		archive := deriveFixture(fx)
		reason := staleness(archive, archive+".derivation.json", wantedDerivation(fx))
		check(fx.name+" derives", reason == "", firstOf(reason, archive))
	}
}

// wantedDerivation is the identity the CURRENT tree gives a fixture: its wire schema and
// the hash of everything of the fixture's that lives in this repository.
func wantedDerivation(fx fixtureSpec) derivation {
	wanted := derivation{
		WireSchemaDigest: pb.SchemaDigest,
		RuntimeSHA:       resolveRuntimeSHA(),
		ScriptSHA256:     map[string]string{fx.script: hashTreeFile(fx.script)},
		SourceSHA256:     map[string]string{},
	}
	for _, src := range fx.sources {
		wanted.SourceSHA256[src] = hashTreeFile(src)
	}
	return wanted
}

func firstOf(reason, fallback string) string {
	if reason != "" {
		return reason
	}
	return fallback
}
