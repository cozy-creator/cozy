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
// checkout it provisioned, exactly as it does for -worker-protocol-fixtures.
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
	wantRepository = "https://github.com/cozy-creator/worker-protocol-v2"
	sourceManifest = vendorDir + "/SOURCE"
	digestValuePfx = "sha256:"
	reVendorRemedy = "re-vendor: copy " + peerDir + "/*.go into protocol/cozy/worker/v1/ and rewrite SOURCE"
)

// vendorManifest is SOURCE parsed into its `key=value` lines.
type vendorManifest struct {
	repository string
	commit     string
	wireMinor  uint32
	digests    map[string]string // file name -> lowercase hex sha256
}

func readVendorManifest(t *testing.T) vendorManifest {
	t.Helper()
	raw, err := os.ReadFile(sourceManifest)
	if err != nil {
		t.Fatalf("vendored worker protocol has no SOURCE manifest: %v", err)
	}
	m := vendorManifest{digests: map[string]string{}}
	for i, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			t.Fatalf("%s:%d: not a key=value line: %q", sourceManifest, i+1, line)
		}
		switch {
		case key == "repository":
			m.repository = value
		case key == "commit":
			m.commit = value
		case key == "wire_minor":
			n, err := strconv.ParseUint(value, 10, 32)
			if err != nil {
				t.Fatalf("%s:%d: wire_minor %q is not a number", sourceManifest, i+1, value)
			}
			m.wireMinor = uint32(n)
		case strings.HasSuffix(key, ".go"):
			digest, ok := strings.CutPrefix(value, digestValuePfx)
			if !ok || len(digest) != 64 {
				t.Fatalf("%s:%d: %s digest is not a sha256: %q", sourceManifest, i+1, key, value)
			}
			m.digests[key] = strings.ToLower(digest)
		default:
			t.Fatalf("%s:%d: unknown manifest key %q", sourceManifest, i+1, key)
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
	m := readVendorManifest(t)
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

// TestVendoredBytesMatchManifest is the always-armed half: the bytes on disk are
// exactly what SOURCE says they are, and SOURCE names exactly the files present.
// This catches a hand-edit of generated code and a half-landed re-vendor that
// adds, drops, or rewrites a file without rewriting the manifest — including a
// retired file left beside its replacement.
func TestVendoredBytesMatchManifest(t *testing.T) {
	m := readVendorManifest(t)
	if len(m.digests) == 0 {
		t.Fatal("SOURCE lists no generated files; the manifest is not guarding anything")
	}

	entries, err := os.ReadDir(vendorDir)
	if err != nil {
		t.Fatalf("read %s: %v", vendorDir, err)
	}
	onDisk := map[string]bool{}
	for _, entry := range entries {
		if !entry.IsDir() && strings.HasSuffix(entry.Name(), ".go") {
			onDisk[entry.Name()] = true
		}
	}

	for name, want := range m.digests {
		if !onDisk[name] {
			t.Errorf("SOURCE lists %s but it is not vendored; %s", name, reVendorRemedy)
			continue
		}
		if got := sha256File(t, filepath.Join(vendorDir, name)); got != want {
			t.Errorf("%s digest mismatch:\n  on disk %s\n  SOURCE  %s\nthe vendored bytes were "+
				"edited without rewriting SOURCE; %s", name, got, want, reVendorRemedy)
		}
	}
	for name := range onDisk {
		if _, listed := m.digests[name]; !listed {
			t.Errorf("%s is vendored but absent from SOURCE; an unlisted generated file is "+
				"unguarded — %s", name, reVendorRemedy)
		}
	}
}

// TestWireMinorMatchesManifest catches the exact shape of the incident this file
// exists for: a re-vendor that half-lands, leaving the compiled-in wire minor and
// the recorded one disagreeing.
func TestWireMinorMatchesManifest(t *testing.T) {
	m := readVendorManifest(t)
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
	m := readVendorManifest(t)
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
