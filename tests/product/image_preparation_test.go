package producttest

import (
	"image"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/inputasset"
	"github.com/cozy-creator/cozy/internal/launch"
)

// Client-side preparation is optional work: a policy this host cannot apply exactly keeps
// the interface usable and sends raw bytes. Only a mistyped member refuses the document.
func TestImagePreparationDescriptorAppliesOnlyExactDecodedPolicies(t *testing.T) {
	decoded := strings.Replace(declaredAssetsInterface, `"parameter":"assets"`, `"parameter":"assets","view":"decoded"`, 1)
	for _, test := range []struct {
		policy, outcome string
	}{
		{`{"profile":"image-fit/1","max_edge":8192,"max_pixels":16777216}`, "prepared"},
		{`{"profile":"image-fit/1","max_edge":64}`, "prepared"},
		{`{"profile":"image-fit/1","max_pixels":256}`, "prepared"},
		{`{"profile":"image-fit/2","max_edge":64}`, "prepared"},
		{`{"profile":"image-fit/1"}`, "raw"},
		{`{"profile":"image-fit/1","max_edge":0}`, "raw"},
		{`{"profile":"image-fit/1","max_pixels":null}`, "raw"},
		{`{"profile":"image-fit/1","max_edge":64,"crop":true}`, "raw"},
		{`{"profile":"image-fit/1","max_edge":true}`, "refused"},
		{`{"profile":"image-fit/1","max_pixels":1.5}`, "refused"},
	} {
		raw := strings.Replace(decoded, `"kind":"image"`, `"kind":"image","prepare":`+test.policy, 1)
		ep, problem := callableOf(t, []byte(raw), "run")
		if (problem != nil) != (test.outcome == "refused") {
			t.Fatalf("policy %s admission: %v", test.policy, problem)
		}
		if problem != nil {
			continue
		}
		if prepared := ep.Assets.Kinds[0].Preparation != nil; prepared != (test.outcome == "prepared") {
			t.Fatalf("policy %s: prepared=%v, want %s", test.policy, prepared, test.outcome)
		}
	}
	policy := `"prepare":{"profile":"image-fit/1","max_edge":64},`
	for name, raw := range map[string]string{
		"raw view":   strings.Replace(declaredAssetsInterface, `"kind":"image"`, policy+`"kind":"image"`, 1),
		"video kind": strings.Replace(strings.Replace(decoded, `"kind":"image"`, policy+`"kind":"image"`, 1), `"kind":"image"`, `"kind":"video"`, 1),
	} {
		iface, problem := launch.DecodePackageInterface([]byte(raw))
		fatal(t, problem)
		ep, problem := iface.Function("run")
		fatal(t, problem)
		if ep.Assets.Kinds[0].Preparation != nil {
			t.Fatalf("%s applied image preparation", name)
		}
	}
}

func TestImageSourceProbeDoesNotHashTheLargeSource(t *testing.T) {
	path := filepath.Join(t.TempDir(), "source.png")
	file, err := os.Create(path)
	must(t, err)
	must(t, png.Encode(file, image.NewRGBA(image.Rect(0, 0, 2, 2))))
	const sourceLength = int64(5) << 30
	must(t, file.Truncate(sourceLength))
	must(t, file.Close())
	length, mime, problem := inputasset.Probe(path)
	fatal(t, problem)
	if length != sourceLength || mime != "image/png" {
		t.Fatalf("header/stat probe changed facts: %d %s", length, mime)
	}
	if _, problem := inputasset.Fingerprint(path, 1024); problem == nil {
		t.Fatal("admitted byte bound was widened by the header probe")
	}
}
