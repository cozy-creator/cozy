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
		"outputs/a222/image": "other-result", ".reservations/a111": "64",
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
	for _, path := range []string{"inputs/a111-payload", "inputs/a111-input-0", "outputs/a111", ".reservations/a111"} {
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

func TestWorkerWriteAndHTTPUploadShareReservedQuota(t *testing.T) {
	base := t.TempDir()
	root := filepath.Join(base, "media")
	for _, dir := range []string{filepath.Join(root, "inputs"), filepath.Join(root, "outputs")} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
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
	authorized := func(req *http.Request) {
		req.Header.Set("Authorization", "Bearer test-token")
	}

	reserve := httptest.NewRequest(http.MethodPost, "/v1/outputs/a111?max_bytes=6", nil)
	authorized(reserve)
	reserved := httptest.NewRecorder()
	s.routes().ServeHTTP(reserved, reserve)
	if reserved.Code != http.StatusCreated {
		t.Fatalf("reserve status = %d, body %s", reserved.Code, reserved.Body.String())
	}

	start := make(chan struct{})
	workerDone := make(chan error, 1)
	uploadCode := make(chan int, 1)
	go func() {
		<-start
		workerDone <- os.WriteFile(filepath.Join(root, "outputs", "a111", "image"),
			[]byte("123456"), 0o644)
	}()
	go func() {
		<-start
		req := httptest.NewRequest(http.MethodPut, "/v1/inputs/blob", bytes.NewReader([]byte("12345")))
		authorized(req)
		response := httptest.NewRecorder()
		s.routes().ServeHTTP(response, req)
		uploadCode <- response.Code
	}()
	close(start)
	if err := <-workerDone; err != nil {
		t.Fatal(err)
	}
	code := <-uploadCode
	if code != http.StatusRequestEntityTooLarge && code != http.StatusInsufficientStorage {
		t.Fatalf("HTTP upload racing the worker = %d, want quota refusal", code)
	}
	if got := s.used(); got != 6 {
		t.Fatalf("accounted quota after direct worker write = %d, want exact 6 B reservation", got)
	}
	if _, err := os.Stat(filepath.Join(root, "inputs", "blob")); !os.IsNotExist(err) {
		t.Fatalf("refused HTTP bytes landed despite the output reservation: %v", err)
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
	if _, err := os.Stat(filepath.Join(root, "inputs", ".blob.staging")); !os.IsNotExist(err) {
		t.Fatalf("shared staging name remains: %v", err)
	}
}
