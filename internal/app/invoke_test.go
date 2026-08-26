package app

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCorruptOutputNeverReplacesFinalSave(t *testing.T) {
	good := []byte("good")
	sum := sha256.Sum256(good)
	digest := "sha256:" + hex.EncodeToString(sum[:])
	tests := []struct {
		name, bytes, errorName string
	}{
		{name: "same length wrong digest", bytes: "evil", errorName: "media_digest_mismatch"},
		{name: "truncated", bytes: "go", errorName: "media_length_mismatch"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			final := filepath.Join(dir, "image.png")
			if err := os.WriteFile(final, []byte("previous verified output"), 0o600); err != nil {
				t.Fatal(err)
			}
			staged, _, _, e := stageVerifiedOutput(final, int64(len(good)), digest, strings.NewReader(tt.bytes))
			if e == nil || e.ErrName() != tt.errorName || staged != "" {
				t.Fatalf("stage corrupt output = %q, %v; want %s", staged, e, tt.errorName)
			}
			data, err := os.ReadFile(final)
			if err != nil || string(data) != "previous verified output" {
				t.Fatalf("corrupt transfer touched final save: %q, %v", data, err)
			}
			if leftovers, _ := filepath.Glob(filepath.Join(dir, ".image.png.staging-*")); len(leftovers) != 0 {
				t.Fatalf("corrupt transfer left staging files: %v", leftovers)
			}
		})
	}
}
