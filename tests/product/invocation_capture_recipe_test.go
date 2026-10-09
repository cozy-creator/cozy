package producttest

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/packagepublish"
)

func TestOldInvocationCaptureIsNotReusedAfterSDKWheelCorrection(t *testing.T) {
	root := t.TempDir()
	current := map[string]packagepublish.SourceStamp{"pyproject.toml": {Size: 12, Modified: 34}}
	old, err := json.Marshal(current)
	must(t, err)
	path := filepath.Join(root, "invocation-source-stats.json")
	must(t, os.WriteFile(path, old, 0600))
	if packagepublish.InvocationSourceUnchanged(root, current) {
		t.Fatal("new command reused a capture generated with old SDK equality pins")
	}
	retained, err := os.ReadFile(path)
	must(t, err)
	if string(retained) != string(old) {
		t.Fatal("cache lookup rewrote an accepted capture")
	}
	fresh := t.TempDir()
	fatal(t, packagepublish.RecordInvocationSource(fresh, current))
	if !packagepublish.InvocationSourceUnchanged(fresh, current) {
		t.Fatal("corrected capture is not reusable")
	}
	changed := map[string]packagepublish.SourceStamp{"pyproject.toml": {Size: 13, Modified: 34}}
	if packagepublish.InvocationSourceUnchanged(fresh, changed) {
		t.Fatal("changed authored source reused a capture")
	}
}
