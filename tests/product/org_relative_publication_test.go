package producttest

// One package source publishes to any Tensorhub under whichever account publishes it. The
// source names no org: its default ladder is org-relative and its same-account dependency
// resolves from the reserved `tensorhub` index. Two stand-in Hubs, one per account, receive
// artifacts that carry their own absolute model refs, index URLs and dependency hashes, and a
// run of each published release resolves the default to its own account's model.

import (
	"archive/zip"
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/records"
)

const orgRelativeDependency = "org-relative-dep"

const orgRelativeApp = `import msgspec
from cozy_runtime.author import App, Loader, Model, uses_components


class Request(msgspec.Struct):
    steps: int = 1


class Result(msgspec.Struct):
    count: int


class H3(Model):
    def load(self, loader: Loader) -> None:
        pass

    @uses_components("video_vae")
    def decode_video(self) -> object:
        return None


app = App()


@app.entrypoint(defaults={"model": [{"gpu": "*", "lane": "minimax@1.0.0-rc.1/fp8-adaln-pruned"}]})
def generate(payload: Request, model: H3) -> Result:
    return Result(payload.steps)
`

// publishingHub is one account's Tensorhub: the ladder stand-in's h3/minimax catalog under that
// org, plus publication and the account's package index serving its own dependency wheel.
type publishingHub struct {
	*ladderHub
	account  string
	versions map[string][]byte // dependency version -> wheel bytes on this Hub
	mu       sync.Mutex
	declared map[string]hub.PackageDeclaredFile
	uploaded map[string][]byte
	registry []hub.PackageRegistryRow
}

func newPublishingHub(t *testing.T, account string, versions map[string][]byte) *publishingHub {
	t.Helper()
	h := &publishingHub{ladderHub: newOrgLadderHub(t, account), account: account, versions: versions,
		declared: map[string]hub.PackageDeclaredFile{}, uploaded: map[string][]byte{}}
	h.ladderHub.mu.Lock()
	h.ladderHub.iface = nil
	h.ladderHub.mu.Unlock()
	mux := h.mux
	mux.HandleFunc("GET /v1/accounts/current", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(hub.Account{Name: account})
	})
	mux.HandleFunc("GET /v1/index/{org}/simple/{distribution}/", func(w http.ResponseWriter, r *http.Request) {
		if r.PathValue("org") != account || r.PathValue("distribution") != orgRelativeDependency {
			http.NotFound(w, r)
			return
		}
		for version, wheel := range versions {
			digest := fmt.Sprintf("%x", sha256.Sum256(wheel))
			filename := orgRelativeWheelName(version)
			fmt.Fprintf(w, `<a href="/v1/index/%s/files/%s/%s#sha256=%s">%s</a><br>`, account, digest, filename, digest, filename)
		}
	})
	mux.HandleFunc("GET /v1/index/{org}/files/{digest}/{filename}", func(w http.ResponseWriter, r *http.Request) {
		for version, wheel := range versions {
			if r.PathValue("org") == account && r.PathValue("filename") == orgRelativeWheelName(version) &&
				r.PathValue("digest") == fmt.Sprintf("%x", sha256.Sum256(wheel)) {
				_, _ = w.Write(wheel)
				return
			}
		}
		http.NotFound(w, r)
	})
	base := "/v1/packages/" + account + "/h3/publish/1.0.0"
	mux.HandleFunc("POST "+base, func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Files []hub.PackageDeclaredFile `json:"files"`
		}
		must(t, json.NewDecoder(r.Body).Decode(&body))
		answer := hub.PackageReleaseDraft{PublicationID: account + "-publication"}
		h.mu.Lock()
		for _, file := range body.Files {
			h.declared[file.Path] = file
			answer.Files = append(answer.Files, hub.PackageFileGrant{PackageDeclaredFile: file,
				Upload: &hub.PackagePresignedUpload{URL: h.server.URL + "/upload?path=" + url.QueryEscape(file.Path)}})
		}
		h.mu.Unlock()
		_ = json.NewEncoder(w).Encode(answer)
	})
	mux.HandleFunc("PUT /upload", func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		must(t, err)
		name := r.URL.Query().Get("path")
		digest := sha256.Sum256(raw)
		h.mu.Lock()
		defer h.mu.Unlock()
		if file := h.declared[name]; int64(len(raw)) != file.Length || "sha256:"+hex.EncodeToString(digest[:]) != file.Digest {
			t.Errorf("%s: uploaded bytes differ from the declaration", name)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		h.uploaded[name] = raw
		w.WriteHeader(http.StatusCreated)
	})
	mux.HandleFunc("POST "+base+"/finalize", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Registry []hub.PackageRegistryRow `json:"registry"`
		}
		must(t, json.NewDecoder(r.Body).Decode(&body))
		h.mu.Lock()
		h.registry = body.Registry
		iface := h.uploaded["metadata/package-interface.json"]
		complete := len(h.uploaded) == len(h.declared)
		h.mu.Unlock()
		if !complete || iface == nil {
			t.Error("finalize before every declared file arrived")
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		h.ladderHub.mu.Lock()
		h.ladderHub.iface = iface
		h.ladderHub.mu.Unlock()
		_ = json.NewEncoder(w).Encode(hub.PackageReleaseCommit{PublicationID: account + "-publication", State: "committed"})
	})
	return h
}

func (h *publishingHub) file(t *testing.T, suffix string) []byte {
	t.Helper()
	h.mu.Lock()
	defer h.mu.Unlock()
	for name, raw := range h.uploaded {
		if strings.HasSuffix(name, suffix) {
			return raw
		}
	}
	t.Fatalf("%s received no %s", h.account, suffix)
	return nil
}

func orgRelativeWheelName(version string) string {
	return strings.ReplaceAll(orgRelativeDependency, "-", "_") + "-" + version + "-py3-none-any.whl"
}

// orgRelativeWheel is a pure dependency wheel; body differs per Hub so its hash names the Hub.
func orgRelativeWheel(t *testing.T, version, body string) []byte {
	t.Helper()
	var out bytes.Buffer
	archive := zip.NewWriter(&out)
	info := strings.ReplaceAll(orgRelativeDependency, "-", "_") + "-" + version + ".dist-info/"
	record := ""
	for _, member := range [][2]string{
		{"org_relative_dep.py", body},
		{info + "METADATA", fmt.Sprintf("Metadata-Version: 2.3\nName: %s\nVersion: %s\n", orgRelativeDependency, version)},
		{info + "WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"},
	} {
		writer, err := archive.Create(member[0])
		must(t, err)
		_, err = writer.Write([]byte(member[1]))
		must(t, err)
		digest := sha256.Sum256([]byte(member[1]))
		record += fmt.Sprintf("%s,sha256=%s,%d\n", member[0], base64.RawURLEncoding.EncodeToString(digest[:]), len(member[1]))
	}
	writer, err := archive.Create(info + "RECORD")
	must(t, err)
	_, err = writer.Write([]byte(record + info + "RECORD,,\n"))
	must(t, err)
	must(t, archive.Close())
	return out.Bytes()
}

// orgRelativeRuntime installs the Runtime under test as the host tool and answers a PATH
// that finds it first.
func orgRelativeRuntime(t *testing.T) string {
	t.Helper()
	wheel, err := filepath.Abs(*privateChildRuntimeWheel)
	must(t, err)
	venv := filepath.Join(t.TempDir(), "runtime")
	for _, args := range [][]string{{"venv", venv, "--python", "3.12"}, {"pip", "install", "--python", filepath.Join(venv, "bin", "python"), wheel}} {
		if out, err := exec.Command("uv", args...).CombinedOutput(); err != nil {
			t.Fatalf("uv %v: %v\n%s", args, err, out)
		}
	}
	return filepath.Join(venv, "bin") + string(os.PathListSeparator) + os.Getenv("PATH")
}

// orgRelativeSource is the one authored tree: no org anywhere, a committed interface read by
// the Runtime under test, and no uv.lock until `cozy package lock` writes one.
func orgRelativeSource(t *testing.T, path string) string {
	t.Helper()
	source := filepath.Join(t.TempDir(), "h3")
	must(t, os.MkdirAll(filepath.Join(source, "metadata"), 0o755))
	for name, body := range map[string]string{
		"h3.py":        orgRelativeApp,
		"package.toml": "[application]\nobject = \"h3:app\"\n",
		"pyproject.toml": `[project]
name = "h3"
version = "1.0.0"
requires-python = ">=3.12,<3.13"
dependencies = ["org-relative-dep>=1.0,<2"]

[project.entry-points."cozy.application"]
default = "h3:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
only-include = ["h3.py"]

[tool.uv.sources]
org-relative-dep = { index = "tensorhub" }
`,
	} {
		must(t, os.WriteFile(filepath.Join(source, name), []byte(body), 0o644))
	}
	runtime := filepath.Join(filepath.SplitList(path)[0], "cozy-runtime")
	raw, err := exec.Command(runtime, "--json", "--dir", source, "describe").Output()
	if err != nil {
		t.Fatalf("describe the org-relative source: %v", err)
	}
	if !bytes.Contains(raw, []byte(`"lane":"minimax@1.0.0-rc.1/fp8-adaln-pruned"`)) {
		t.Fatalf("the Runtime did not record the org-relative lane as written:\n%s", raw)
	}
	must(t, os.WriteFile(filepath.Join(source, packagepublish.CommittedInterfacePath), raw, 0o644))
	return source
}

func runCozyIn(t *testing.T, dir, root, path string, args ...string) (int, string) {
	t.Helper()
	cmd := exec.Command("/usr/bin/nice", append([]string{"-n", "19", cozyBin}, args...)...)
	cmd.Dir = dir
	cmd.Env = childEnv(t, root, "PATH="+path)
	data, _ := cmd.CombinedOutput()
	code := 0
	if cmd.ProcessState != nil {
		code = cmd.ProcessState.ExitCode()
	}
	return code, string(data)
}

func TestOneSourcePublishesToEachHubAsItsAccount(t *testing.T) {
	integration(t)
	if *privateChildRuntimeWheel == "" {
		t.Skip("requires a Runtime wheel that admits org-relative lanes: -child-runtime-wheel")
	}
	path := orgRelativeRuntime(t)
	source := orgRelativeSource(t, path)
	authoredPyproject, err := os.ReadFile(filepath.Join(source, "pyproject.toml"))
	must(t, err)
	hubs := []*publishingHub{
		newPublishingHub(t, "paul", map[string][]byte{"1.0.0": orgRelativeWheel(t, "1.0.0", "HUB = 'paul'\n")}),
		newPublishingHub(t, "fidika", map[string][]byte{"1.0.0": orgRelativeWheel(t, "1.0.0", "HUB = 'fidika'\n")}),
	}
	roots := []string{ladderRoot(t, hubs[0].ladderHub), ladderRoot(t, hubs[1].ladderHub)}
	for _, root := range roots {
		t.Cleanup(func() { _, _ = runCozy(t, root, "down", "--all") })
	}

	// The author locks once, against whichever Hub is current: here paul's.
	if code, out := runCozyIn(t, source, roots[0], path, "package", "lock", "--json"); code != 0 {
		t.Fatalf("cozy package lock: %d\n%s", code, out)
	}
	committedLock, err := os.ReadFile(filepath.Join(source, "uv.lock"))
	must(t, err)
	if !strings.Contains(string(committedLock), hubs[0].server.URL+"/v1/index/paul/simple/") {
		t.Fatalf("the lock does not resolve paul's index on paul's Hub:\n%s", committedLock)
	}

	var projectWheels [][]byte
	for i, h := range hubs {
		if code, out := runCozyIn(t, source, roots[i], path, "package", "publish", "--json"); code != 0 {
			t.Fatalf("publish as %s: %d\n%s", h.account, code, out)
		}
		index := h.server.URL + "/v1/index/" + h.account + "/simple/"
		digest := fmt.Sprintf("%x", sha256.Sum256(h.versions["1.0.0"]))

		// The interface names this account's model; nothing org-relative reaches the Hub.
		iface, problem := launch.DecodePackageInterface(h.file(t, "metadata/package-interface.json"))
		fatal(t, problem)
		lane := iface.Entrypoints[0].Models[0].DefaultLadder[0].Lane
		if lane != h.account+"/minimax@1.0.0-rc.1/fp8-adaln-pruned" {
			t.Fatalf("%s published default lane %q", h.account, lane)
		}
		// pyproject and lock are the bound copy for this Hub and account.
		pyproject := string(h.file(t, "pyproject.toml"))
		if !strings.Contains(pyproject, "name = \"tensorhub\"") || !strings.Contains(pyproject, fmt.Sprintf("url = %q", index)) ||
			!strings.Contains(pyproject, "explicit = true") {
			t.Fatalf("%s published pyproject names no account index %s:\n%s", h.account, index, pyproject)
		}
		lock := string(h.file(t, "uv.lock"))
		if !strings.Contains(lock, `registry = "`+index+`"`) || !strings.Contains(lock, "sha256:"+digest) {
			t.Fatalf("%s published lock is not bound to %s and its wheel %s:\n%s", h.account, index, digest, lock)
		}
		for _, other := range hubs {
			if other != h && strings.Contains(lock+pyproject, other.server.URL) {
				t.Fatalf("%s's release names %s's Hub", h.account, other.account)
			}
		}
		h.mu.Lock()
		registry := append([]hub.PackageRegistryRow(nil), h.registry...)
		h.mu.Unlock()
		if len(registry) != 1 || registry[0].SHA256 != digest ||
			registry[0].URL != h.server.URL+"/v1/index/"+h.account+"/files/"+digest+"/"+orgRelativeWheelName("1.0.0") {
			t.Fatalf("%s registry rows: %+v", h.account, registry)
		}
		projectWheels = append(projectWheels, h.file(t, "-py3-none-any.whl"))
		t.Logf("%s: lane %s; index %s; dependency %s", h.account, lane, index, registry[0].URL)

		// A run of the published release resolves the default to this account's model.
		key := "org-relative-" + h.account
		code, out := runCozy(t, roots[i], "run", h.account+"/h3/generate", "steps=1", "--rental-only", "--json", "--idempotency-key", key)
		store, problem := records.Open(filepath.Join(roots[i], "creator.sqlite"))
		fatal(t, problem)
		row, problem := store.RequestByIdempotencyKey(key)
		store.Close()
		fatal(t, problem)
		if row == nil || len(row.Models) != 1 || row.Models[0].Model != h.account+"/minimax" || row.Models[0].Release != "1.0.0-rc.1" {
			t.Fatalf("%s's run did not resolve its own model: %d %s row=%+v", h.account, code, out, row)
		}
		t.Logf("%s: run %s resolved %s@%s", h.account, row.ID, row.Models[0].Model, row.Models[0].Release)
	}
	if !bytes.Equal(projectWheels[0], projectWheels[1]) {
		t.Fatal("the project wheel differs between accounts; only metadata may")
	}
	after, err := os.ReadFile(filepath.Join(source, "pyproject.toml"))
	must(t, err)
	lockAfter, err := os.ReadFile(filepath.Join(source, "uv.lock"))
	must(t, err)
	if !bytes.Equal(after, authoredPyproject) || !bytes.Equal(lockAfter, committedLock) {
		t.Fatal("publication changed the authored pyproject or lock")
	}

	// An editable install binds its owned snapshot to the current caller's index, and an
	// editable run resolves the org-relative default to that caller's model.
	for i, h := range hubs {
		if code, out := runCozyIn(t, t.TempDir(), roots[i], path, "package", "install", source, "--editable", "--json"); code != 0 {
			t.Fatalf("editable install as %s: %d\n%s", h.account, code, out)
		}
		installed := activeInstall(t, roots[i], "local/h3")
		python := filepath.Join(installed.Dir, "venv", "bin", "python")
		hubName, err := exec.Command(python, "-c", "import org_relative_dep; print(org_relative_dep.HUB)").Output()
		if err != nil || strings.TrimSpace(string(hubName)) != h.account {
			t.Fatalf("%s's editable environment installed the dependency from %q: %v", h.account, hubName, err)
		}
		key := "org-relative-editable-" + h.account
		code, out := runCozyIn(t, t.TempDir(), roots[i], path, "run", "local/h3/generate", "steps=1", "--rental-only", "--json", "--idempotency-key", key)
		store, problem := records.Open(filepath.Join(roots[i], "creator.sqlite"))
		fatal(t, problem)
		row, problem := store.RequestByIdempotencyKey(key)
		store.Close()
		fatal(t, problem)
		// The run resolves before it buys; the stand-in's later purchase outcome is not the proof.
		if row == nil || len(row.Models) != 1 || row.Models[0].Model != h.account+"/minimax" || row.Models[0].Release != "1.0.0-rc.1" {
			t.Fatalf("%s's editable run did not resolve its caller's model: %d %s row=%+v", h.account, code, out, row)
		}
		t.Logf("%s editable: dependency from its own Hub; run %s resolved %s", h.account, row.ID, row.Models[0].Model)
	}

	// A Hub whose account lacks the locked version refuses rather than resolving another.
	bob := newPublishingHub(t, "bob", map[string][]byte{"1.1.0": orgRelativeWheel(t, "1.1.0", "HUB = 'bob'\n")})
	bobRoot := ladderRoot(t, bob.ladderHub)
	t.Cleanup(func() { _, _ = runCozy(t, bobRoot, "down", "--all") })
	code, out := runCozyIn(t, source, bobRoot, path, "package", "publish", "--json")
	if code == 0 || !strings.Contains(out, "account_index_lock_drift") || !strings.Contains(out, "org-relative-dep 1.0.0") {
		t.Fatalf("publishing with an unpublished locked dependency did not refuse drift: %d\n%s", code, out)
	}
	t.Logf("bob: %s", strings.TrimSpace(out))
}
