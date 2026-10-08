package accountauth

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"

	"github.com/cozy-creator/cozy/internal/config"
)

// KnownOrigins names saved logins, including unnamed --tensorhub URLs. This reads
// only the public origin header; authentication still validates its own credential.
func KnownOrigins(home string) []string {
	directory := filepath.Join(home, "auth")
	entries, _ := os.ReadDir(directory)
	var origins []string
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".json" {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(directory, entry.Name()))
		var header struct {
			Hub string `json:"hub"`
		}
		if err != nil || json.Unmarshal(raw, &header) != nil {
			continue
		}
		origin, problem := config.HubOrigin(header.Hub)
		if problem != nil || origin != header.Hub {
			continue
		}
		sum := sha256.Sum256([]byte(origin))
		if entry.Name() == hex.EncodeToString(sum[:])+".json" {
			origins = append(origins, origin)
		}
	}
	sort.Strings(origins)
	return origins
}
