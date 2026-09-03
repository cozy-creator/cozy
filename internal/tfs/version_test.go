package tfs

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
)

// presentBinary writes an executable that behaves like a tfs build and Opens against it,
// so every assertion runs the real resolve -> handshake -> store-init path.
func presentBinary(t *testing.T, script string) (*Tool, *exit.Error) {
	t.Helper()
	bin := filepath.Join(t.TempDir(), "tfs")
	if err := os.WriteFile(bin, []byte("#!/bin/sh\n"+script+"\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	cfg := config.Config{Tfs: bin, TfsSource: "test"}
	return Open(cfg, home.Layout{CAS: filepath.Join(t.TempDir(), "cas")})
}

func TestOpenAcceptsTheBuiltForVersion(t *testing.T) {
	digest := "sha256:" + strings.Repeat("ab", 32)
	tool, e := presentBinary(t, fmt.Sprintf(`if [ "$1" = version ]; then echo "tfs %s %s"; fi; exit 0`, BuiltFor, digest))
	if e != nil {
		t.Fatalf("matching binary refused: %s", e.Message)
	}
	if tool.Version != BuiltFor || tool.BuildDigest != digest {
		t.Fatalf("handshake recorded %q %q", tool.Version, tool.BuildDigest)
	}
}

func TestOpenRefusesAMismatchedBinary(t *testing.T) {
	_, e := presentBinary(t, `echo "tfs 9.9.9 sha256:`+strings.Repeat("cd", 32)+`"`)
	if e == nil {
		t.Fatal("a mismatched tensorfs build was accepted")
	}
	if e.Name != "tfs_version_skew" || e.Code != exit.Structural {
		t.Fatalf("refusal %s (exit %d)", e.ErrName(), e.Code)
	}
	if !strings.Contains(e.Message, "9.9.9") || !strings.Contains(e.Message, BuiltFor) {
		t.Fatalf("refusal names neither side: %s", e.Message)
	}
}

func TestOpenRefusesABinaryWithoutAVersionSurface(t *testing.T) {
	// A pre-tfs-051 build: `tfs version` prints usage and exits 2.
	_, e := presentBinary(t, `echo "usage: see the header" >&2; exit 2`)
	if e == nil {
		t.Fatal("a versionless tensorfs build was accepted")
	}
	if e.Name != "tfs_version_unreadable" || e.Code != exit.Structural {
		t.Fatalf("refusal %s (exit %d)", e.ErrName(), e.Code)
	}
}
