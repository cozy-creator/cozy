package producttest

import (
	"crypto/sha256"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/packagepublish"
)

func TestPrivateCaptureDoesNotApplyPublicationAccountPolicy(t *testing.T) {
	helper := prebuiltProjectWheel(t, t.TempDir(), "hub-fixture", "py3-none-any", "weightless:app")
	data, err := os.ReadFile(helper)
	must(t, err)
	digest := fmt.Sprintf("%x", sha256.Sum256(data))
	file := "/v1/index/paul/files/" + digest + "/" + filepath.Base(helper)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == file {
			w.Header().Set("Content-Length", fmt.Sprint(len(data)))
			w.Write(data)
			return
		}
		w.Header().Set("Content-Type", "text/html")
		fmt.Fprintf(w, "<a href=%q>%s</a>", file+"#sha256="+digest, filepath.Base(helper))
	}))
	defer server.Close()
	project := fixtureTree(t, fixturePyproject("hub-fixture>=1.0")+fmt.Sprintf("\n[tool.uv.sources]\nhub-fixture={index='tensorhub'}\n[[tool.uv.index]]\nname='tensorhub'\nurl=%q\nexplicit=true\n", server.URL+"/v1/index/paul/simple/"), minimalFixtureLock)
	application, err := os.ReadFile(filepath.Join("testdata", "weightless", "weightless.py"))
	must(t, err)
	must(t, os.WriteFile(filepath.Join(project, "src", "cozy_fixture_package", "__init__.py"), application, 0600))
	command := exec.CommandContext(t.Context(), "uv", "lock", "--directory", project)
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("lock Hub fixture: %v\n%s", err, output)
	}
	// The previous StagePrepared entry point enters publication policy with no
	// publishing account. Its refusal is valid for that path, not for private work.
	previous, problem := packagepublish.PrepareLocalFrom(project)
	fatal(t, problem)
	problem = previous.Build(t.Context())
	previous.Close()
	if problem == nil || problem.Name != "registry_dependency_index_refused" {
		t.Fatalf("base build did not reproduce account-policy refusal: %v", problem)
	}
	pack, problem := packagepublish.PrepareLocalFrom(project)
	fatal(t, problem)
	defer pack.Close()
	fatal(t, pack.BuildForCapture(t.Context()))
	fatal(t, pack.CaptureUnpublishedClosure(t.Context(), "cozy-fixture-package==1.0.0\nhub-fixture==1.0.0", nil, "3.12.12"))
	if len(pack.DependencyWheels) != 1 || len(pack.DependencyRequirements) != 0 || pack.DependencyPackages["hub-fixture"] != "paul/hub-fixture" {
		t.Fatalf("private Hub dependency did not retain exact custody: %+v %s", pack.DependencyWheels, pack.DependencyRequirements)
	}
	if strings.Contains(string(pack.DependencyRequirements), "127.0.0.1") {
		t.Fatal("remote closure depends on the author's loopback Hub")
	}
}
