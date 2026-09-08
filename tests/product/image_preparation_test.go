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

func TestImagePreparationDescriptorIsClosedAndDecodedOnly(t *testing.T) {
	decoded := strings.Replace(declaredAssetsInterface, `"parameter":"assets"`, `"parameter":"assets","view":"decoded"`, 1)
	for _, test := range []struct {
		policy string
		valid  bool
	}{
		{`{"profile":"image-fit/1","max_edge":8192,"max_pixels":16777216}`, true},
		{`{"profile":"image-fit/1","max_edge":64}`, true},
		{`{"profile":"image-fit/1","max_pixels":256}`, true},
		{`{"profile":"unknown","max_edge":64}`, false},
		{`{"profile":"image-fit/1"}`, false},
		{`{"profile":"image-fit/1","max_edge":0}`, false},
		{`{"profile":"image-fit/1","max_edge":true}`, false},
		{`{"profile":"image-fit/1","max_pixels":null}`, false},
		{`{"profile":"image-fit/1","max_pixels":1.5}`, false},
		{`{"profile":"image-fit/1","max_edge":64,"crop":true}`, false},
	} {
		raw := strings.Replace(decoded, `"kind":"image"`, `"kind":"image","prepare":`+test.policy, 1)
		iface, problem := launch.DecodePackageInterface([]byte(raw))
		if (problem == nil) != test.valid {
			t.Fatalf("policy %s admission: %v", test.policy, problem)
		}
		if test.valid {
			ep, problem := iface.Function("run")
			fatal(t, problem)
			if ep.Assets.Kinds[0].Preparation == nil || ep.Assets.Kinds[0].Preparation.Profile != "image-fit/1" {
				t.Fatal("preparation policy disappeared")
			}
		}
	}
	policy := `"prepare":{"profile":"image-fit/1","max_edge":64},`
	raw := strings.Replace(declaredAssetsInterface, `"kind":"image"`, policy+`"kind":"image"`, 1)
	if _, problem := launch.DecodePackageInterface([]byte(raw)); problem == nil {
		t.Fatal("raw-view Assets admitted image preparation")
	}
	wrongKind := strings.Replace(strings.Replace(decoded, `"kind":"image"`, policy+`"kind":"image"`, 1), `"kind":"image"`, `"kind":"video"`, 1)
	if _, problem := launch.DecodePackageInterface([]byte(wrongKind)); problem == nil {
		t.Fatal("video kind admitted image preparation")
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
	if _, _, _, problem := inputasset.Fingerprint(path, 1024); problem == nil {
		t.Fatal("admitted byte bound was widened by the header probe")
	}
}
