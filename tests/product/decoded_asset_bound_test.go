package producttest

import (
	"fmt"
	"testing"

	"github.com/cozy-creator/cozy/internal/launch"
)

func TestDeclaredAssetDecodedBoundUsesClosedPositiveInteger(t *testing.T) {
	for _, value := range []string{"4096", "9007199254740991", "0", "-1", "1.5", "true", `"4096"`, "null", "9007199254740992"} {
		t.Run(value, func(t *testing.T) {
			raw := []byte(fmt.Sprintf(`{"format":"cozy.package.interface/1","application":"fixture:app","jobs":[],"entrypoints":[{"name":"run","models":[],"request":{"fields":[{"name":"image","type":{"asset":"image"},"asset_bound":{"max_bytes":1024,"max_decoded_bytes":%s,"media_types":["image/png"]}}]},"result":{"fields":[]}}]}`, value))
			entry, problem := launch.DecodePackageInterface(raw)
			valid := value == "4096" || value == "9007199254740991"
			if (problem == nil) != valid {
				t.Fatalf("decoded bound %s: %v", value, problem)
			}
			if valid && entry.Entrypoints[0].Request.Fields[0].AssetBound.MaxDecodedBytes <= 0 {
				t.Fatal("decoded byte limit was discarded")
			}
		})
	}
}
