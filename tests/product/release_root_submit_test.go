package producttest

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
)

// installedHere records a published release as `cozy package install` leaves it, so a run
// naming a rental these tests cannot reach reads the release's interface here.
func installedHere(t *testing.T, root, hubURL, pkg, release string) {
	t.Helper()
	response, err := http.Get(hubURL + "/v1/packages/" + pkg + "/releases/" + release)
	must(t, err)
	var detail struct {
		PackageInterface json.RawMessage `json:"package_interface"`
	}
	must(t, json.NewDecoder(response.Body).Decode(&detail))
	response.Body.Close()
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	sum := sha256.Sum256([]byte(pkg + "@" + release))
	id := fmt.Sprintf("%x", sum[:8])
	installed := records.PackageInstall{ID: id, Package: pkg, Major: 1, Version: release,
		SourceKind: "tensorhub", SourceRef: pkg + "@" + release, Verified: true, Dir: layout.InstallDir(id)}
	iface, err := canonical.NormalizeJCS(detail.PackageInterface)
	must(t, err)
	must(t, os.MkdirAll(filepath.Dir(launch.PackageInterfacePath(installed.Dir)), 0o700))
	must(t, os.WriteFile(launch.PackageInterfacePath(installed.Dir), iface, 0o444))
	_, problem = store.Activate(installed)
	fatal(t, problem)
}
