package producttest

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestDashedModelFlagsReuseExistingResolver(t *testing.T) {
	root, mu, posts, _, _ := runModelCatalog(t)
	base := []string{"--json", "run", "proof/quantize/quantize", "steps=7", "--rental-only", "--dry-run", "--publish-to", "proof/output"}
	canonical := append(append([]string{}, base[:4]...), append([]string{"model.dits=proof/source@1.0.0/bf16", "model.shared=proof/source@1.0.0/bf16"}, base[4:]...)...)
	code, encoded := runCozy(t, root, canonical...)
	var want map[string]any
	if code != 0 || json.Unmarshal([]byte(encoded), &want) != nil {
		t.Fatalf("canonical dry run failed: %d %s", code, encoded)
	}
	for _, args := range [][]string{
		append(append([]string{}, base...), "--model.dits=proof/source@1.0.0/bf16", "--model.shared=proof/source@1.0.0/bf16"),
		append([]string{"--json", "run", "--model.dits=proof/source@1.0.0/bf16", "--model.shared=proof/source@1.0.0/bf16"}, base[2:]...),
	} {
		code, encoded := runCozy(t, root, args...)
		var got map[string]any
		if code != 0 || json.Unmarshal([]byte(encoded), &got) != nil || !reflect.DeepEqual(got, want) {
			t.Fatalf("dashed override changed resolver meaning: %d %s", code, encoded)
		}
	}
	for _, extra := range []string{"--model.other=proof/source@1.0.0/bf16", "--model.dits=proof/source@1.0.0/bf16"} {
		code, out := runCozy(t, root, append(append([]string{}, canonical...), extra)...)
		if code == 0 || strings.Contains(out, "unknown flag") {
			t.Fatalf("unknown/duplicate model slot bypassed the existing resolver: %d %s", code, out)
		}
	}
	for _, args := range [][]string{
		{"--json", "model", "info", "proof/source", "--model.other=bad"},
		{"--json", "--fields", "run", "model", "info", "proof/source", "--model.other=bad"},
		{"--json", "run", "watch", "1", "--model.other=bad"},
		{"--json", "run", "proof/quantize/quantize", "--describe", "--stream"},
	} {
		code, out := runCozy(t, root, args...)
		if code == 0 || !strings.Contains(out, "unknown flag") {
			t.Fatalf("unsupported flag was reinterpreted: %d %s", code, out)
		}
	}
	if _, err := os.Stat(filepath.Join(root, "daemon.lock")); !os.IsNotExist(err) {
		t.Fatal("metadata-only model flags proof started a daemon")
	}
	mu.Lock()
	defer mu.Unlock()
	if len(*posts) != 0 {
		t.Fatal("metadata-only model flags proof attempted a rental")
	}
}
