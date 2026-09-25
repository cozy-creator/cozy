package producttest

import (
	"encoding/json"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/resultfiles"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func TestRuntime023ForwardedPNGSchemaRecovery(t *testing.T) {
	captured := json.RawMessage(`{"fields":[{"asset_bound":{"max_bytes":33554432,"max_decoded_bytes":20971520,"media_types":["image/png"]},"name":"value","type":{"asset":"image"}}]}`)
	legacy := json.RawMessage(`{"fields":[{"asset_bound":{"max_bytes":33554432,"media_types":["image/png"]},"name":"value","type":{"asset":"image"}}]}`)
	normalized, err := canonical.NormalizeJCS(legacy)
	must(t, err)
	envelope := &pb.ResultEnvelope{ResultSchemaDigest: canonical.Digest(normalized), InlineResult: []byte(`{"value":{"asset_ref":"sha256:` + strings.Repeat("a", 64) + `","digest":"sha256:` + strings.Repeat("a", 64) + `","kind":"image","media_type":"image/png","size_bytes":100}}`)}
	if launch.ValidateMachineResult(captured, envelope) == nil {
		t.Fatal("strict schema validation ignored decoded bound")
	}
	projected, maximum, ok := launch.Runtime023PNGResultSchema(captured, envelope, "0.18.23")
	if !ok || maximum != 20971520 {
		t.Fatal("exact released projection was not recognized")
	}
	fatal(t, launch.ValidateMachineResult(projected, envelope))
	for _, version := range []string{"", "0.18.22", "0.18.24", "0.18.23+unverified"} {
		if _, _, ok := launch.Runtime023PNGResultSchema(captured, envelope, version); ok {
			t.Fatalf("unqualified version %s", version)
		}
	}
	for _, changed := range []string{
		strings.Replace(string(captured), "33554432", "33554431", 1),
		strings.Replace(string(captured), "image/png", "image/jpeg", 1),
		strings.Replace(string(captured), `"value"`, `"other"`, 1),
		strings.Replace(string(captured), "20971520", "0", 1),
	} {
		if _, _, ok := launch.Runtime023PNGResultSchema(json.RawMessage(changed), envelope, "0.18.23"); ok {
			t.Fatal("unrelated schema drift accepted")
		}
	}
	envelope.ResultSchemaDigest = make([]byte, 32)
	if _, _, ok := launch.Runtime023PNGResultSchema(captured, envelope, "0.18.23"); ok {
		t.Fatal("unknown result schema accepted")
	}
}

func TestRecoveredPNGDecodedBoundRejectsCompressedOversize(t *testing.T) {
	path := filepath.Join(t.TempDir(), "image.png")
	file, err := os.Create(path)
	must(t, err)
	must(t, png.Encode(file, image.NewRGBA(image.Rect(0, 0, 512, 512))))
	must(t, file.Close())
	info, err := os.Stat(path)
	must(t, err)
	if info.Size() >= 512*512*3 {
		t.Fatal("fixture did not compress below decoded bytes")
	}
	fatal(t, resultfiles.VerifyPNGDecodedBound(path, 512*512*3))
	for _, limit := range []int64{0, info.Size(), 512*512*3 - 1} {
		if resultfiles.VerifyPNGDecodedBound(path, limit) == nil {
			t.Fatalf("oversized decoded PNG accepted at %d", limit)
		}
	}
	must(t, os.WriteFile(path, []byte("not a PNG"), 0600))
	if resultfiles.VerifyPNGDecodedBound(path, 512*512*3) == nil {
		t.Fatal("invalid image accepted")
	}
}
