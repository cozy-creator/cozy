package producttest

import (
	"fmt"
	"testing"

	"github.com/cozy-creator/cozy/internal/launch"
)

// The decoded-byte bound is Runtime-owned: any integer reading keeps the interface usable,
// and only a mistyped member refuses.
func TestDeclaredAssetDecodedBoundIsRuntimeOwned(t *testing.T) {
	for _, value := range []string{"4096", "9007199254740991", "0", "-1", "null", "1.5", "true", `"4096"`} {
		t.Run(value, func(t *testing.T) {
			raw := []byte(fmt.Sprintf(`{"format":"cozy.package.interface/1","application":"fixture:app","jobs":[],"entrypoints":[{"name":"run","models":[],"request":{"fields":[{"name":"image","type":{"asset":"image"},"asset_bound":{"max_bytes":1024,"max_decoded_bytes":%s,"media_types":["image/png"]}}]},"result":{"fields":[]}}]}`, value))
			entry, problem := launch.DecodePackageInterface(raw)
			mistyped := value == "1.5" || value == "true" || value == `"4096"`
			if (problem != nil) != mistyped {
				t.Fatalf("decoded bound %s: %v", value, problem)
			}
			if value == "4096" && entry.Entrypoints[0].Request.Fields[0].AssetBound.MaxDecodedBytes != 4096 {
				t.Fatal("decoded byte limit was discarded")
			}
		})
	}
}
