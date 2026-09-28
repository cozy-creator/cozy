package producttest

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/hub"
)

const calleePackage = "proof/child"

type calleeFacts struct {
	iface  []byte
	locked string
}

// publishCalleeRelease replaces proof/h3@1.0.0 with a torch-free release whose CPU
// `long_form` job has no model of its own and whose lock pins proof/child@0.1.0 from the
// org index. The callee's `generate` entrypoint holds the ladder's model slot.
func publishCalleeRelease(t *testing.T, h *ladderHub) map[string]calleeFacts {
	t.Helper()
	return publishCalleeReleaseOf(t, h, `{"application":"h3:app","entrypoints":[],"format":"cozy.package.interface/1","jobs":[{"models":[],"name":"long_form","publishes":false,"request":{"fields":[]},"result":{"fields":[]}}]}`)
}

// publishCalleeReleaseOf publishes the same pair with the root's own interface.
func publishCalleeReleaseOf(t *testing.T, h *ladderHub, root string) map[string]calleeFacts {
	t.Helper()
	childWheel := []byte("child wheel")
	childDigest := strings.TrimPrefix(mustSpell(childWheel), "sha256:")
	rootIface, err := canonical.NormalizeJCS([]byte(root))
	must(t, err)
	childIface, err := canonical.NormalizeJCS([]byte(`{"application":"child:app","entrypoints":[{"invocable":{"context":"ctx","defaults":{},"enum_members":{},"export":"generate","module":"child","parameters":["steps"],"type_names":{}},"models":[{"class":"H3","component_use":{"condition_text":["text_encoder"],"decode_video":["video_vae"],"sample_fl2va":["fl2va_dit"]},"path":"generate.models.model"}],"name":"generate","request":{"fields":[{"name":"steps","type":"int"}]},"result":{"fields":[]}}],"format":"cozy.package.interface/1","jobs":[]}`))
	must(t, err)
	index := h.server.URL + "/v1/index/proof/simple/"
	rootLock := "version = 1\nrequires-python = \">=3.12\"\n\n" +
		"[[package]]\nname = \"child\"\nversion = \"0.1.0\"\nsource = { registry = \"" + index + "\" }\n" +
		"wheels = [{ url = \"" + h.server.URL + "/v1/index/proof/files/" + childDigest + "/child-0.1.0-py3-none-any.whl\", hash = \"sha256:" + childDigest + "\" }]\n\n" +
		"[[package]]\nname = \"h3\"\nversion = \"1.0.0\"\nsource = { editable = \".\" }\ndependencies = [{ name = \"child\" }]\n"
	release := func(name, version string, iface []byte, pyproject, lock string) (hub.PackageReleaseDetail, hub.PackageDownloadPlan) {
		exact := func(raw []byte) hub.ExactDocument {
			return hub.ExactDocument{CanonicalBytes: raw, Digest: mustSpell(raw), Length: int64(len(raw))}
		}
		var detail hub.PackageReleaseDetail
		detail.PackageInterface = iface
		detail.Release.Release = version
		detail.Release.PackageInterfaceDigest = mustSpell(iface)
		detail.Release.PackageInterfaceLength = int64(len(iface))
		detail.ExecutionRequirements = []string{"cozy-runtime>=0.2.25"}
		wheel := []byte(name + " wheel")
		filename := name + "-" + version + "-py3-none-any.whl"
		return detail, hub.PackageDownloadPlan{Release: version,
			PackageConfig:    exact([]byte("[application]\nobject = \"" + name + ":app\"\n")),
			PackageInterface: exact(iface),
			Pyproject:        exact([]byte(pyproject)),
			UVLock:           exact([]byte(lock)),
			Downloads: []hub.PackageInstallDownload{{Kind: "project_wheel", Path: filename,
				Distribution: name, Version: version, Digest: mustSpell(wheel), Length: int64(len(wheel)),
				Tags: []string{"py3-none-any"}, ImportRoots: []string{name}}}}
	}
	rootDetail, rootPlan := release("h3", "1.0.0", rootIface,
		"[project]\nname = \"h3\"\nversion = \"1.0.0\"\nrequires-python = \">=3.12\"\ndependencies = [\"child==0.1.0\"]\n", rootLock)
	childDetail, childPlan := release("child", "0.1.0", childIface,
		"[project]\nname = \"child\"\nversion = \"0.1.0\"\nrequires-python = \">=3.12\"\ndependencies = []\n",
		"version = 1\nrequires-python = \">=3.12\"\n\n[[package]]\nname = \"child\"\nversion = \"0.1.0\"\nsource = { editable = \".\" }\n")
	pin := func(pkg, release string) string {
		return string(testLockedRequirements(pkg, release))
	}
	facts := map[string]calleeFacts{
		ladderPackage: {iface: rootIface, locked: pin(ladderPackage, "1.0.0") + "child==0.1.0 --hash=sha256:" + childDigest + "\n"},
		calleePackage: {iface: childIface, locked: pin(calleePackage, "0.1.0")},
	}
	fallback := h.server.Config.Handler
	h.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body any
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/v1/packages/proof/h3/releases/1.0.0":
			body = rootDetail
		case r.Method == http.MethodPost && r.URL.Path == "/v1/packages/proof/h3/download" && r.URL.Query().Get("release") == "1.0.0":
			body = rootPlan
		case r.Method == http.MethodGet && r.URL.Path == "/v1/packages/"+calleePackage:
			body = hub.PackageCard{Package: hub.Resource{Org: "proof", Name: "child"}, Releases: []hub.ReleaseSummary{{Release: "0.1.0"}}}
		case r.Method == http.MethodGet && r.URL.Path == "/v1/packages/"+calleePackage+"/releases/0.1.0":
			body = childDetail
		case r.Method == http.MethodPost && r.URL.Path == "/v1/packages/"+calleePackage+"/download" && r.URL.Query().Get("release") == "0.1.0":
			body = childPlan
		case r.Method == http.MethodGet && r.URL.Path == "/v1/packages/"+calleePackage+"/bindings":
			h.mu.Lock()
			body = map[string]any{"bindings": append([]hub.PackageBindingRow{}, h.bindings...)}
			h.mu.Unlock()
		case r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/locked-requirements"):
			// The temporary prepared path reads each release's lock (proto-062 R2).
			pkg := strings.TrimPrefix(strings.TrimSuffix(r.URL.Path, "/locked-requirements"), "/v1/packages/")
			selected, known := facts[pkg[:strings.LastIndex(pkg, "/releases/")]]
			if !known {
				http.NotFound(w, r)
				return
			}
			_, _ = w.Write([]byte(selected.locked))
			return
		default:
			fallback.ServeHTTP(w, r)
			return
		}
		_ = json.NewEncoder(w).Encode(body)
	})
	return facts
}
