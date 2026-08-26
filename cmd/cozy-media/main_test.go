package main

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/cozy-creator/cozy-creator-v2/internal/secret"
)

func TestDropAttemptRemovesOnlyOwnedMedia(t *testing.T) {
	root := t.TempDir()
	token := secret.New("test-token")
	tokens := filepath.Join(root, "tokens")
	if err := os.WriteFile(tokens, []byte(secret.HashLine(token)+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	for path, data := range map[string]string{
		"inputs/a111-payload": "payload", "inputs/a111-input-0": "asset",
		"inputs/a222-payload": "other", "outputs/a111/image": "result",
		"outputs/a222/image": "other-result",
	} {
		full := filepath.Join(root, path)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(data), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	s := &server{opt: options{root: root, tokens: tokens}}
	request := httptest.NewRequest(http.MethodDelete, "/v1/attempts/a111", nil)
	request.Header.Set("Authorization", "Bearer test-token")
	response := httptest.NewRecorder()
	s.routes().ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("DELETE status = %d, body %s", response.Code, response.Body.String())
	}
	for _, path := range []string{"inputs/a111-payload", "inputs/a111-input-0", "outputs/a111"} {
		if _, err := os.Stat(filepath.Join(root, path)); !os.IsNotExist(err) {
			t.Fatalf("owned media %s remains after DELETE: %v", path, err)
		}
	}
	for _, path := range []string{"inputs/a222-payload", "outputs/a222/image"} {
		if _, err := os.Stat(filepath.Join(root, path)); err != nil {
			t.Fatalf("unrelated media %s was touched: %v", path, err)
		}
	}
}

func TestConcurrentUploadsShareOneQuotaDecision(t *testing.T) {
	base := t.TempDir()
	root := filepath.Join(base, "media")
	if err := os.MkdirAll(filepath.Join(root, "inputs"), 0o755); err != nil {
		t.Fatal(err)
	}
	token := secret.New("test-token")
	tokens := filepath.Join(base, "tokens")
	if err := os.WriteFile(tokens, []byte(secret.HashLine(token)+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	s := &server{opt: options{root: root, tokens: tokens, quota: 10, maxBody: 10}}
	if err := s.reloadTokens(); err != nil {
		t.Fatal(err)
	}
	start := make(chan struct{})
	codes := make(chan int, 2)
	var wg sync.WaitGroup
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			req := httptest.NewRequest(http.MethodPut, "/v1/inputs/blob", bytes.NewReader([]byte("123456")))
			req.Header.Set("Authorization", "Bearer test-token")
			response := httptest.NewRecorder()
			s.routes().ServeHTTP(response, req)
			codes <- response.Code
		}()
	}
	close(start)
	wg.Wait()
	close(codes)
	created, refused := 0, 0
	for code := range codes {
		if code == http.StatusCreated {
			created++
		} else if code == http.StatusRequestEntityTooLarge || code == http.StatusInsufficientStorage {
			refused++
		}
	}
	if created != 1 || refused != 1 || s.used() > s.opt.quota {
		t.Fatalf("concurrent uploads: created=%d refused=%d used=%d quota=%d",
			created, refused, s.used(), s.opt.quota)
	}
	if _, err := os.Stat(filepath.Join(root, "inputs", "blob.staging")); !os.IsNotExist(err) {
		t.Fatalf("shared staging name remains: %v", err)
	}
}
