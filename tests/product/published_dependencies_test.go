package producttest

import (
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/install"
)

func TestPublishedDependenciesCaptureDirectWheelAndExactVersionPins(t *testing.T) {
	const digest = "763658f4804cd6496c12604ee69d0a6fcd9fe2ba8336fa2a7ea9f2d1657fd070"
	const wheelURL = "https://serving.example/v1/index/paul/files/" + digest + "/reference_image-0.1.1-py3-none-any.whl"
	lock := []byte(`version = 1
[[package]]
name = "reference-image"
version = "0.1.1"
source = { registry = "http://127.0.0.1:1/v1/index/paul/simple/" }
wheels = [{url = "http://127.0.0.1:1/v1/index/paul/files/` + digest + `/reference_image-0.1.1-py3-none-any.whl", hash = "sha256:` + digest + `"}]
`)
	direct := "reference-image @ " + wheelURL + " --hash=sha256:" + digest + "\n"
	version := "reference-image==0.1.1 --hash=sha256:" + digest + "\n"
	for _, row := range []string{direct, version} {
		selected, problem := install.PublishedDependencies("paul/minimax-h3", lock, []byte("--index-url https://pypi.org/simple\n"+row))
		fatal(t, problem)
		if len(selected) != 1 || selected[0].Package != "paul/reference-image" || selected[0].Version != "0.1.1" || len(selected[0].Digests) != 1 || !selected[0].Digests["sha256:"+digest] {
			t.Fatalf("prepared callable dependency lost from capture: %#v", selected)
		}
	}
	for name, rows := range map[string]string{
		"wrong digest":     strings.ReplaceAll(direct, "--hash=sha256:"+digest, "--hash=sha256:"+strings.Repeat("a", 64)),
		"duplicate direct": direct + direct,
		"two row forms":    direct + version,
	} {
		if _, problem := install.PublishedDependencies("paul/minimax-h3", lock, []byte(rows)); problem == nil {
			t.Fatalf("%s admitted ambiguous or unlocked callable", name)
		}
	}
	for name, source := range map[string][]byte{
		"PyPI homonym": []byte(strings.ReplaceAll(string(lock), "http://127.0.0.1:1/v1/index/paul/simple/", "https://pypi.org/simple")),
		"foreign org":  []byte(strings.ReplaceAll(string(lock), "/index/paul/simple/", "/index/other/simple/")),
	} {
		selected, problem := install.PublishedDependencies("paul/minimax-h3", source, []byte(direct))
		fatal(t, problem)
		if len(selected) != 0 {
			t.Fatalf("%s conferred callable authority", name)
		}
	}
	selected, problem := install.PublishedDependencies("paul/minimax-h3", lock, []byte(strings.ReplaceAll(version, "==0.1.1", "==0.1.2")))
	fatal(t, problem)
	if len(selected) != 0 {
		t.Fatal("another version conferred callable authority")
	}
}
