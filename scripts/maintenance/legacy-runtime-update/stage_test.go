package main

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/machines"
)

func TestStageVerifiedWheelOnlyWritesTheStagingEndpoint(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	body := []byte("verified wheel fixture")
	digest := sha256.Sum256(body)
	sha := hex.EncodeToString(digest[:])
	name := "cozy_runtime-0.18.92-cp312-abi3-manylinux_2_28_x86_64.whl"
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, body, 0o600); err != nil {
		t.Fatal(err)
	}
	var paths []string
	badReceipt := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.Method+" "+r.URL.Path)
		if r.Method != http.MethodPut || r.URL.Path != "/v1/machine/runtime/wheels/"+name || !strings.HasPrefix(r.Header.Get("Authorization"), "Cozy-Cap ") {
			t.Error("staging used an unexpected endpoint or omitted its capability")
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		got, err := io.ReadAll(r.Body)
		if err != nil || string(got) != string(body) {
			t.Errorf("staged body changed: %v", err)
		}
		answer := stagedWheel{File: name, SHA256: sha, Length: int64(len(body))}
		if badReceipt {
			answer.SHA256 = strings.Repeat("0", 64)
		}
		_ = json.NewEncoder(w).Encode(answer)
	}))
	defer server.Close()
	client := &machines.Maintenance{Base: server.URL, Machine: "stage-proof", Public: public,
		Client: server.Client(), Sign: func(message []byte) []byte { return ed25519.Sign(private, message) }}
	want := pair{"0.18.92", "0.3.78"}
	for _, wrong := range []string{"invalid=" + path, strings.Repeat("0", 64) + "=" + path} {
		if _, err := stageWheel(t.Context(), client, wrong, want); err == nil {
			t.Fatal("an unverified wheel was accepted")
		}
	}
	if _, err := stageWheel(t.Context(), client, sha+"="+path, pair{"0.18.93", "0.3.78"}); err == nil {
		t.Fatal("a wheel for another Runtime was accepted")
	}
	if len(paths) != 0 {
		t.Fatal("an invalid candidate reached the machine")
	}
	got, err := stageWheel(t.Context(), client, sha+"="+path, want)
	if err != nil || got.File != name || got.SHA256 != sha || got.Length != int64(len(body)) {
		t.Fatalf("verified staging failed: %+v %v", got, err)
	}
	badReceipt = true
	if _, err := stageWheel(t.Context(), client, sha+"="+path, want); err == nil {
		t.Fatal("a mismatched machine receipt was accepted")
	}
	if len(paths) != 2 {
		t.Fatalf("staging sent extra requests: %v", paths)
	}
}
