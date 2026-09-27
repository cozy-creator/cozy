// Vendored cozy.worker.v1 drift guard.
//
// Cozy does not own the worker protocol; it vendors the generated Go for a
// pinned upstream commit. The failure this file exists to prevent is THIS
// repository's own: it sat at wire minor 15 while the schema and every other
// consumer had moved to 16, and nothing said so, because the only check that
// could have said so needed a credential CI does not always have.
//
// So the digest half needs no network, no secret and no checkout: it always
// runs, in every `go test ./...`. The peer half is opt-in through the same flag
// idiom canonical_test.go already uses -- the environment is read once at the
// entrypoint here, never in a test -- and when it is asked for and cannot run,
// it FAILS. Nothing here ever calls t.Skip.
package producttest

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// Point this at a local worker-protocol clone to compare real bytes. CI passes the
// checkout it provisioned. The vendored trees below need no such checkout to be
// checked; this only adds upstream provenance on top of them.
var workerProtocolRepo = flag.String("worker-protocol-repo", "",
	"a worker-protocol checkout; the vendored bytes are compared against it")

// Set by any wiring that MEANS to run the peer comparison. If it is set and the
// comparison cannot run, that is a failure, not a skip.
var requireWorkerProtocolPeer = flag.Bool("require-worker-protocol-peer", false,
	"fail instead of reporting when -worker-protocol-repo is absent")

const (
	// Relative to this package, because #661 keeps every Go test in tests/product.
	vendorDir = "../../protocol/cozy/worker/v1"
	// Where the generated Go lives inside a worker-protocol checkout.
	peerDir        = "gen/go/cozy/worker/v1"
	wantRepository = "https://github.com/cozy-creator/worker-protocol"
	sourceManifest = vendorDir + "/SOURCE"
	// worker-protocol's canonical corpus and selected byte-transport fixtures,
	// vendored so the cross-language identity fence needs no token.
	corpusDir = "testdata/worker-protocol"
	// Where that corpus lives inside a worker-protocol checkout.
	peerCorpusDir  = "fixtures"
	corpusManifest = corpusDir + "/SOURCE"
	digestValuePfx = "sha256:"
	reVendorRemedy = "re-vendor: copy " + peerDir + "/*.go into protocol/cozy/worker/v1/ and rewrite SOURCE"
	reCorpusRemedy = "re-vendor: copy the canonical/ and red/ documents and MANIFEST.json from " +
		peerCorpusDir + "/ into " + corpusDir + "/ and rewrite its SOURCE"
)

// vendorManifest is SOURCE parsed into its `key=value` lines.
type vendorManifest struct {
	repository string
	commit     string
	wireMinor  uint32
	digests    map[string]string // file name -> lowercase hex sha256
}

// isVendoredFileKey keeps the parser strict: a manifest key is either one of the
// three header fields or a relative path to a file this repo vendors. Anything
// else is a typo, and a typo that parses is a digest nobody checks.
func isVendoredFileKey(key string) bool {
	if key == "" || strings.HasPrefix(key, "/") || strings.Contains(key, "..") {
		return false
	}
	switch filepath.Ext(key) {
	case ".go", ".json", ".bin":
		return true
	}
	return false
}

func readVendorManifest(t *testing.T, manifestPath string) vendorManifest {
	t.Helper()
	raw, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("%s is missing: %v", manifestPath, err)
	}
	m := vendorManifest{digests: map[string]string{}}
	for i, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			t.Fatalf("%s:%d: not a key=value line: %q", manifestPath, i+1, line)
		}
		switch {
		case key == "repository":
			m.repository = value
		case key == "commit":
			m.commit = value
		case key == "wire_minor":
			n, err := strconv.ParseUint(value, 10, 32)
			if err != nil {
				t.Fatalf("%s:%d: wire_minor %q is not a number", manifestPath, i+1, value)
			}
			m.wireMinor = uint32(n)
		case isVendoredFileKey(key):
			digest, ok := strings.CutPrefix(value, digestValuePfx)
			if !ok || len(digest) != 64 {
				t.Fatalf("%s:%d: %s digest is not a sha256: %q", manifestPath, i+1, key, value)
			}
			m.digests[key] = strings.ToLower(digest)
		default:
			t.Fatalf("%s:%d: unknown manifest key %q", manifestPath, i+1, key)
		}
	}
	return m
}

func sha256File(t *testing.T, path string) string {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:])
}

// TestVendorManifestIsExact checks the manifest itself before anything trusts it.
func TestVendorManifestIsExact(t *testing.T) {
	m := readVendorManifest(t, sourceManifest)
	if m.repository != wantRepository {
		t.Errorf("SOURCE repository = %q, want %q", m.repository, wantRepository)
	}
	if len(m.commit) != 40 {
		t.Errorf("SOURCE commit %q is not a full 40-hex commit", m.commit)
	}
	if _, err := hex.DecodeString(m.commit); err != nil {
		t.Errorf("SOURCE commit %q is not hex", m.commit)
	}
	if m.wireMinor == 0 {
		t.Error("SOURCE records no wire_minor; a re-vendor that forgets it must not look clean")
	}
}

// checkTreeMatchesManifest is the always-armed comparison, shared by both vendored
// trees: the bytes on disk are exactly what the manifest says they are, and the
// manifest names exactly the files present. This catches a hand-edit, and a
// half-landed re-vendor that adds, drops, or rewrites a file without rewriting the
// manifest — including a retired file left beside its replacement.
func checkTreeMatchesManifest(t *testing.T, root string, m vendorManifest, remedy string) {
	t.Helper()
	if len(m.digests) == 0 {
		t.Fatalf("%s/SOURCE lists no files; the manifest is not guarding anything", root)
	}

	onDisk := map[string]bool{}
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		rel, relErr := filepath.Rel(root, path)
		if relErr != nil {
			return relErr
		}
		rel = filepath.ToSlash(rel)
		if rel == "SOURCE" {
			return nil
		}
		onDisk[rel] = true
		return nil
	})
	if err != nil {
		t.Fatalf("walk %s: %v", root, err)
	}

	for name, want := range m.digests {
		if !onDisk[name] {
			t.Errorf("%s/SOURCE lists %s but it is not vendored; %s", root, name, remedy)
			continue
		}
		if got := sha256File(t, filepath.Join(root, filepath.FromSlash(name))); got != want {
			t.Errorf("%s/%s digest mismatch:\n  on disk %s\n  SOURCE  %s\nthe vendored bytes "+
				"were changed without rewriting SOURCE; %s", root, name, got, want, remedy)
		}
	}
	for name := range onDisk {
		if _, listed := m.digests[name]; !listed {
			t.Errorf("%s/%s is vendored but absent from SOURCE; an unlisted file is "+
				"unguarded — %s", root, name, remedy)
		}
	}
}

// TestVendoredBytesMatchManifest guards the generated bindings.
func TestVendoredBytesMatchManifest(t *testing.T) {
	checkTreeMatchesManifest(t, vendorDir, readVendorManifest(t, sourceManifest), reVendorRemedy)
}

// TestVendoredCorpusMatchesManifest guards the frozen canonical corpus.
//
// The corpus is vendored so TestCanonicalDocuments runs in every `go test ./...`
// with no token and no sibling checkout — the cross-language identity check is
// the one whose failure is least visible any other way, and it spent weeks
// skipping on every CI run instead. The obvious objection to vendoring it is that
// it changes on nearly every schema commit and vendored-diff.sh does not look at
// fixtures, so it would become a NEW silent-drift surface. This is the answer:
// the corpus carries its own SOURCE in the same key=value shape as the bindings',
// naming the worker-protocol commit it was taken from and a sha256 per file, and
// a stale or hand-touched corpus is a loud failure here rather than a quiet pass.
func TestVendoredCorpusMatchesManifest(t *testing.T) {
	checkTreeMatchesManifest(t, corpusDir, readVendorManifest(t, corpusManifest), reCorpusRemedy)
}

// TestCorpusAgreesWithBindings closes the triangle. The compiled binding, the
// binding manifest, and the corpus all name a wire minor, and a re-vendor that
// moves one without the others is exactly the half-landed bump this file exists
// to catch. The corpus's own MANIFEST.json is the third witness: it is written by
// worker-protocol, not by us, so it cannot be edited into agreement here.
func TestCorpusAgreesWithBindings(t *testing.T) {
	bindings := readVendorManifest(t, sourceManifest)
	corpus := readVendorManifest(t, corpusManifest)
	if bindings.commit != corpus.commit {
		t.Errorf("the bindings are vendored from %s but the corpus from %s; re-vendor both "+
			"from one worker-protocol commit", bindings.commit, corpus.commit)
	}
	// The corpus may trail the binding within its supported range; a binding older
	// than the corpus is the stale half-landed bump this test exists to catch.
	inRange := func(minor uint32) bool { return minor >= pb.MinCompatibleWireMinor && minor <= pb.WireMinor }
	if !inRange(corpus.wireMinor) {
		t.Errorf("%s records wire_minor = %d outside the compiled binding's range %d..%d",
			corpusManifest, corpus.wireMinor, pb.MinCompatibleWireMinor, pb.WireMinor)
	}
	var frozen struct {
		WireMinor uint32 `json:"wire_minor"`
	}
	raw, err := os.ReadFile(filepath.Join(corpusDir, "MANIFEST.json"))
	if err != nil {
		t.Fatalf("read the frozen MANIFEST.json: %v", err)
	}
	if err := json.Unmarshal(raw, &frozen); err != nil {
		t.Fatalf("parse the frozen MANIFEST.json: %v", err)
	}
	if frozen.WireMinor != corpus.wireMinor || !inRange(frozen.WireMinor) {
		t.Errorf("the frozen corpus was written at wire minor %d; SOURCE records %d and the "+
			"compiled binding speaks %d..%d; %s", frozen.WireMinor, corpus.wireMinor,
			pb.MinCompatibleWireMinor, pb.WireMinor, reCorpusRemedy)
	}
}

// TestWireMinorMatchesManifest catches the exact shape of the incident this file
// exists for: a re-vendor that half-lands, leaving the compiled-in wire minor and
// the recorded one disagreeing.
func TestWireMinorMatchesManifest(t *testing.T) {
	m := readVendorManifest(t, sourceManifest)
	if pb.WireMinor != m.wireMinor {
		t.Errorf("compiled WireMinor = %d but SOURCE records wire_minor = %d.\n"+
			"The vendored code and its manifest disagree about the wire level this build "+
			"speaks; %s", pb.WireMinor, m.wireMinor, reVendorRemedy)
	}
}

// peerGit runs a read-only git command in the peer checkout. Every call is a
// local object read -- rev-parse and show -- so nothing here reaches a network or
// a credential, and the child keeps the ambient environment rather than this test
// reading one (boundaries.md: the environment is read once, at the entrypoint).
func peerGit(t *testing.T, repo string, args ...string) (string, error) {
	t.Helper()
	out, err := exec.Command("git", append([]string{"-C", repo}, args...)...).Output()
	return string(out), err
}

// TestVendoredMatchesUpstream is the peer half. It needs a worker-protocol
// checkout and does two things a digest cannot: it proves the vendored bytes are
// really the pinned commit's bytes, and it reports when upstream has MOVED —
// which is the only way to catch a consumer that forgot to re-vendor.
//
// It is opt-in. It is never silently skipped: if the comparison is asked for and
// cannot run, the test fails.
func TestVendoredMatchesUpstream(t *testing.T) {
	m := readVendorManifest(t, sourceManifest)
	repo := *workerProtocolRepo

	if repo == "" {
		if *requireWorkerProtocolPeer {
			t.Fatal("-require-worker-protocol-peer is set but -worker-protocol-repo is empty: " +
				"the upstream comparison was asked for and cannot run. Pass a worker-protocol " +
				"checkout, or drop the require flag.")
		}
		t.Log("upstream comparison not wired (-worker-protocol-repo unset): the digest and " +
			"wire-minor checks above still ran. Pass a worker-protocol checkout to compare " +
			"real bytes.")
		return
	}

	// A wrong path must fail. Silently skipping on an unreachable fixture is
	// precisely how the stale-wire-minor incident stayed hidden.
	if _, err := peerGit(t, repo, "rev-parse", "--git-dir"); err != nil {
		t.Fatalf("-worker-protocol-repo=%q is not a git checkout: %v", repo, err)
	}
	if _, err := os.Stat(filepath.Join(repo, peerDir)); err != nil {
		t.Fatalf("-worker-protocol-repo=%q has no %s/; it is not a worker-protocol checkout",
			repo, peerDir)
	}

	// 1. The vendored bytes ARE the pinned commit's bytes.
	pinnedMissing := false
	for name, want := range m.digests {
		blob, err := peerGit(t, repo, "show", m.commit+":"+peerDir+"/"+name)
		if err != nil {
			t.Errorf("cannot read %s at pinned commit %s: %v (fetch the checkout?)",
				name, m.commit[:12], err)
			pinnedMissing = true
			continue
		}
		sum := sha256.Sum256([]byte(blob))
		if got := hex.EncodeToString(sum[:]); got != want {
			t.Errorf("%s does not match upstream at the pinned commit %s:\n  vendored %s\n"+
				"  upstream %s", name, m.commit[:12], want, got)
		}
	}
	if pinnedMissing {
		return
	}

	// 2. Has upstream moved past the pin? This is the drift detection.
	head, err := peerGit(t, repo, "rev-parse", "HEAD")
	if err != nil {
		t.Fatalf("cannot resolve the checkout's HEAD: %v", err)
	}
	head = strings.TrimSpace(head)
	if head == m.commit {
		t.Logf("vendored code is current with %s at %s", wantRepository, head[:12])
		return
	}

	var drifted []string
	for name := range m.digests {
		blob, err := peerGit(t, repo, "show", head+":"+peerDir+"/"+name)
		if err != nil {
			drifted = append(drifted, name+" (gone upstream)")
			continue
		}
		sum := sha256.Sum256([]byte(blob))
		if hex.EncodeToString(sum[:]) != m.digests[name] {
			drifted = append(drifted, name)
		}
	}
	if len(drifted) == 0 {
		t.Logf("checkout is at %s, SOURCE pins %s, but the generated bytes are identical",
			head[:12], m.commit[:12])
		return
	}

	upstreamMinor := "unknown"
	if blob, err := peerGit(t, repo, "show", head+":"+peerDir+"/wire_version.go"); err == nil {
		for _, line := range strings.Split(blob, "\n") {
			if _, after, ok := strings.Cut(line, "const WireMinor uint32 = "); ok {
				upstreamMinor = strings.TrimSpace(after)
			}
		}
	}
	t.Errorf("VENDORED WORKER PROTOCOL IS STALE.\n"+
		"  vendored: %s (wire minor %d)\n"+
		"  upstream: %s (wire minor %s)\n"+
		"  drifted:  %s\n"+
		"%s",
		m.commit[:12], pb.WireMinor, head[:12], upstreamMinor,
		strings.Join(drifted, ", "), reVendorRemedy)
}

// TestVendoredCorpusMatchesUpstream is the corpus's peer half, and the reason
// vendoring it does not create a silent-drift surface. The digest half above
// proves the corpus has not been touched since it was copied; this proves it is
// the copy of the commit its SOURCE names, and reports when upstream has frozen a
// newer one. Opt-in through the same flag, and never skipped.
func TestVendoredCorpusMatchesUpstream(t *testing.T) {
	m := readVendorManifest(t, corpusManifest)
	repo := *workerProtocolRepo

	if repo == "" {
		if *requireWorkerProtocolPeer {
			t.Fatal("-require-worker-protocol-peer is set but -worker-protocol-repo is empty: " +
				"the corpus comparison was asked for and cannot run.")
		}
		t.Log("upstream corpus comparison not wired (-worker-protocol-repo unset): the digest " +
			"and wire-minor checks above still ran, and TestCanonicalDocuments ran against " +
			"the vendored corpus.")
		return
	}
	if _, err := peerGit(t, repo, "rev-parse", "--git-dir"); err != nil {
		t.Fatalf("-worker-protocol-repo=%q is not a git checkout: %v", repo, err)
	}

	pinnedMissing := false
	for name, want := range m.digests {
		blob, err := peerGit(t, repo, "show", m.commit+":"+peerCorpusDir+"/"+name)
		if err != nil {
			t.Errorf("cannot read %s at pinned commit %s: %v (fetch the checkout?)",
				name, m.commit[:12], err)
			pinnedMissing = true
			continue
		}
		sum := sha256.Sum256([]byte(blob))
		if got := hex.EncodeToString(sum[:]); got != want {
			t.Errorf("corpus %s does not match upstream at the pinned commit %s:\n"+
				"  vendored %s\n  upstream %s", name, m.commit[:12], want, got)
		}
	}
	if pinnedMissing {
		return
	}

	head, err := peerGit(t, repo, "rev-parse", "HEAD")
	if err != nil {
		t.Fatalf("cannot resolve the checkout's HEAD: %v", err)
	}
	head = strings.TrimSpace(head)
	if head == m.commit {
		t.Logf("vendored corpus is current with %s at %s", wantRepository, head[:12])
		return
	}

	var drifted []string
	for name, want := range m.digests {
		blob, err := peerGit(t, repo, "show", head+":"+peerCorpusDir+"/"+name)
		if err != nil {
			drifted = append(drifted, name+" (gone upstream)")
			continue
		}
		sum := sha256.Sum256([]byte(blob))
		if hex.EncodeToString(sum[:]) != want {
			drifted = append(drifted, name)
		}
	}
	if len(drifted) == 0 {
		t.Logf("checkout is at %s, the corpus SOURCE pins %s, but the frozen bytes are identical",
			head[:12], m.commit[:12])
		return
	}
	t.Errorf("VENDORED CANONICAL CORPUS IS STALE.\n"+
		"  vendored: %s\n  upstream: %s\n  drifted:  %s\n%s",
		m.commit[:12], head[:12], strings.Join(drifted, ", "), reCorpusRemedy)
}
