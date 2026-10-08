package accountauth

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestKnownOriginsUsesOriginBoundCredentialHeaders(t *testing.T) {
	home := t.TempDir()
	if got := KnownOrigins(home); len(got) != 0 {
		t.Fatalf("empty home discovered credentials: %v", got)
	}
	directory := filepath.Join(home, "auth")
	if err := os.Mkdir(directory, 0o700); err != nil {
		t.Fatal(err)
	}
	write := func(name, origin string) {
		t.Helper()
		raw, _ := json.Marshal(map[string]any{"hub": origin, "private_key": "never needed for discovery"})
		if err := os.WriteFile(filepath.Join(directory, name), raw, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	name := func(origin string) string {
		sum := sha256.Sum256([]byte(origin))
		return hex.EncodeToString(sum[:]) + ".json"
	}
	write(name("https://b.example"), "https://b.example")
	write(name("https://a.example"), "https://a.example")
	write(name("https://wrong.example"), "https://other.example")
	write("temporary.json", "https://unbound.example")
	write("invalid.json", "not an origin")
	if got := KnownOrigins(home); !reflect.DeepEqual(got, []string{"https://a.example", "https://b.example"}) {
		t.Fatalf("wrong origins or non-origin data escaped: %v", got)
	}
}
