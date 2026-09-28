package hub

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
)

// A committed release's documents never change, so one read serves every later submission
// of it: they are kept under the local root, per hub. Yanking changes no document, and the
// newest-release question (the package card) is never kept.
func (c *Client) releaseCachePath(kind string, ref Ref, release, variant string) string {
	if c.releases == "" || release == "" {
		return ""
	}
	origin := sha256.Sum256([]byte(c.base))
	name := sha256.Sum256([]byte(kind + "\x00" + ref.String() + "\x00" + release + "\x00" + variant))
	return filepath.Join(c.releases, hex.EncodeToString(origin[:8]), hex.EncodeToString(name[:16])+".json")
}

func loadRelease(path string, out any) bool {
	if path == "" {
		return false
	}
	raw, err := os.ReadFile(path)
	return err == nil && json.Unmarshal(raw, out) == nil
}

// storeRelease is best effort: a document that cannot be kept is read again next time.
func storeRelease(path string, value any) {
	if path == "" {
		return
	}
	raw, err := json.Marshal(value)
	if err != nil || os.MkdirAll(filepath.Dir(path), 0o700) != nil {
		return
	}
	temporary := path + ".tmp"
	if os.WriteFile(temporary, raw, 0o600) == nil {
		_ = os.Rename(temporary, path)
	}
}
